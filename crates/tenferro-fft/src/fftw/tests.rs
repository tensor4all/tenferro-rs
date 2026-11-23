//! Planner thread counts of tenferro plans and of other FFTW callers that plan
//! concurrently in the same process.
use super::*;
use std::ffi::{c_char, CStr};

type SprintPlan = unsafe extern "C" fn(*const c_void) -> *mut c_char;
type Free = unsafe extern "C" fn(*mut c_void);

// Long enough that FFTW's estimate picks a threaded solver whenever the planner
// thread count allows one.
const LEN: usize = 1 << 20;

fn load(threads: usize) -> Option<Fftw> {
    let directory = std::env::var_os("TENFERRO_TEST_FFTW_DIR").map(PathBuf::from);
    let threads = NonZeroUsize::new(threads).unwrap();
    let result = match directory.as_deref() {
        Some(directory) => Fftw::load_from(directory, FftwPlanning::Estimate, threads),
        None => Fftw::load(FftwPlanning::Estimate, threads),
    };
    match result {
        Ok(fftw) => Some(fftw),
        Err(error @ FftwError::Load { .. })
            if directory.is_none() && std::env::var_os("TENFERRO_TEST_REQUIRE_FFTW").is_none() =>
        {
            eprintln!("FFTW integration requires a runtime: {error}");
            None
        }
        Err(error) => panic!("{error}"),
    }
}

/// FFTW's description of a plan, for example `(dft-thr-ct-dit-x3/...)`.
fn describe(api: &Api, plan: *mut c_void) -> String {
    // SAFETY: both symbols have these C signatures in every FFTW 3 release
    // this provider loads.
    let (sprint, free) = unsafe {
        (
            *api._library.get::<SprintPlan>(b"fftw_sprint_plan").unwrap(),
            *api._library.get::<Free>(b"fftw_free").unwrap(),
        )
    };
    // SAFETY: `plan` is a live plan of this library; FFTW returns a
    // NUL-terminated string allocated with `fftw_malloc`.
    unsafe {
        let text = sprint(plan);
        let description = CStr::from_ptr(text).to_string_lossy().into_owned();
        free(text.cast());
        description
    }
}

/// Thread counts of the threaded stages in a plan description: each threaded
/// solver prints a token such as `dft-thr-vrank>=1-x3/1`. The outermost stage
/// has the planner's count; nested stages may have a share of it.
fn thread_counts(description: &str) -> Vec<usize> {
    description
        .split(|c: char| c.is_whitespace() || c == '(' || c == ')')
        .filter(|token| token.contains("-thr-"))
        .map(|token| {
            let count = &token[token.rfind("-x").expect(token) + 2..];
            let digits = count.split('/').next().unwrap();
            digits.parse().expect(token)
        })
        .collect()
}

fn tenferro_plan(fftw: &Fftw, threads: c_int) -> String {
    let plan =
        LanePlan::<f64>::new(fftw.0.clone(), LEN, FftOperation::C2cForward, threads).unwrap();
    describe(&fftw.0.f64, plan.handle.as_ptr())
}

/// A plan created through the FFTW API directly, as another library would.
fn foreign_plan(api: &Api, buffer: &mut [Complex<f64>]) -> String {
    let n = c_int::try_from(buffer.len()).unwrap();
    let p = buffer.as_mut_ptr().cast();
    // SAFETY: an in-place plan over a private buffer of the given length.
    let plan = unsafe { (api.complex)(n, p, p, -1, UNALIGNED | (1 << 6)) };
    assert!(!plan.is_null());
    let description = describe(api, plan);
    // SAFETY: the plan was created above and is destroyed once.
    unsafe { (api.destroy)(plan) };
    description
}

/// Set the ambient planner thread count the way another FFTW caller does,
/// holding the planner lock so that the setter does not race with tenferro
/// plans created by other tests in this binary.
fn set_ambient(threads: ThreadApi, count: c_int) -> c_int {
    before_planning();
    // SAFETY: FFTW's threads are initialized and the planner lock is held.
    let previous = unsafe { (threads.get)() };
    // SAFETY: as above.
    unsafe { (threads.set)(count) };
    after_planning();
    previous
}

#[test]
fn single_threaded_plans_ignore_an_ambient_thread_count() {
    let (Some(threaded), Some(single)) = (load(3), load(1)) else {
        return;
    };
    let threads = threaded.0.f64.threads.unwrap();
    assert!(
        single.0.f64.threads.is_some(),
        "the installation that has a thread API for the threaded provider has one for the single-threaded provider"
    );
    let previous = set_ambient(threads, 3);
    let positive = tenferro_plan(&threaded, 3);
    let single_plan = tenferro_plan(&single, 1);
    let lane_plan = tenferro_plan(&threaded, 1);
    set_ambient(threads, previous);
    let positive = thread_counts(&positive);
    assert_eq!(positive.iter().max(), Some(&3), "{positive:?}");
    assert!(thread_counts(&single_plan).is_empty(), "{single_plan}");
    assert!(thread_counts(&lane_plan).is_empty(), "{lane_plan}");
}

#[test]
fn concurrent_foreign_plans_keep_their_own_thread_count() {
    let Some(provider) = load(2) else {
        return;
    };
    let api = provider.0.f64.clone();
    let threads = api.threads.unwrap();
    let previous = set_ambient(threads, 3);
    let rounds = 40;
    let foreign = std::thread::spawn({
        let api = api.clone();
        move || {
            let mut buffer = vec![Complex::new(0.0, 0.0); LEN];
            (0..rounds)
                .map(|_| foreign_plan(&api, &mut buffer))
                .collect::<Vec<_>>()
        }
    });
    let mut own = Vec::new();
    for _ in 0..rounds {
        own.push((2, tenferro_plan(&provider, 2)));
        own.push((1, tenferro_plan(&provider, 1)));
    }
    let foreign = foreign.join().unwrap();
    set_ambient(threads, previous);
    for description in &foreign {
        let counts = thread_counts(description);
        assert_eq!(counts.iter().max(), Some(&3), "{counts:?}");
    }
    for (expected, description) in &own {
        let counts = thread_counts(description);
        match expected {
            1 => assert!(counts.is_empty(), "{counts:?}"),
            _ => assert_eq!(counts.iter().max(), Some(expected), "{counts:?}"),
        }
    }
}
