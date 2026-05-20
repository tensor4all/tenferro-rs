#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Statistics
using ITensors

const DEFAULT_D = 2
const DEFAULT_CHIS = [1, 2, 4, 8, 16, 32]
const DEFAULT_WARMUP_SECONDS = 1.0
const DEFAULT_MEASUREMENT_SECONDS = 2.0
const DEFAULT_MIN_SAMPLES = 10

function usage()
    println("""
Usage: julia --project=<env-with-ITensors> scripts/bench-itensors-pairwise-contraction.jl [options]

Options:
  --d N                         Physical dimension (default: $(DEFAULT_D))
  --chis LIST                   Comma-separated bond dimensions (default: $(join(DEFAULT_CHIS, ",")))
  --warm-up-time SECONDS        Warm-up time after the first JIT call (default: $(DEFAULT_WARMUP_SECONDS))
  --measurement-time SECONDS    Measurement time per case (default: $(DEFAULT_MEASUREMENT_SECONDS))
  --min-samples N               Minimum samples per case (default: $(DEFAULT_MIN_SAMPLES))
  --blas-threads N              Julia BLAS threads (default: 1)
  --help                        Show this help text
""")
end

function parse_args(args)
    opts = Dict{String, Any}(
        "d" => DEFAULT_D,
        "chis" => copy(DEFAULT_CHIS),
        "warm_up_time" => DEFAULT_WARMUP_SECONDS,
        "measurement_time" => DEFAULT_MEASUREMENT_SECONDS,
        "min_samples" => DEFAULT_MIN_SAMPLES,
        "blas_threads" => 1,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help"
            usage()
            exit(0)
        elseif arg == "--d"
            i += 1
            opts["d"] = parse(Int, args[i])
        elseif arg == "--chis"
            i += 1
            opts["chis"] = parse.(Int, split(args[i], ","))
        elseif arg == "--warm-up-time"
            i += 1
            opts["warm_up_time"] = parse(Float64, args[i])
        elseif arg == "--measurement-time"
            i += 1
            opts["measurement_time"] = parse(Float64, args[i])
        elseif arg == "--min-samples"
            i += 1
            opts["min_samples"] = parse(Int, args[i])
        elseif arg == "--blas-threads"
            i += 1
            opts["blas_threads"] = parse(Int, args[i])
        else
            error("unknown argument: $arg")
        end
        i += 1
    end

    return opts
end

function complex_value(idx::Int, seed::Int)::ComplexF64
    real = ((idx * 17 + seed * 13 + 3) % 97) / 97 - 0.5
    imag = ((idx * 29 + seed * 7 + 5) % 89) / 89 - 0.5
    return ComplexF64(real, imag)
end

function deterministic_itensor(inds_tuple::Tuple, seed::Int)::ITensor
    dims = map(dim, inds_tuple)
    data = Vector{ComplexF64}(undef, prod(dims))
    @inbounds for pos in eachindex(data)
        data[pos] = complex_value(pos - 1, seed)
    end
    return ITensor(reshape(data, dims...), inds_tuple...)
end

function build_fixtures(d::Int, chi::Int)
    site = Index(d, "Site")
    left = Index(chi, "Link,left")
    right_bra = Index(chi, "Link,bra,right")
    right_ket = Index(chi, "Link,ket,right")
    next_ket = Index(chi, "Link,ket,next")

    bra_first = deterministic_itensor((site, right_bra), 1)
    ket_first = deterministic_itensor((site, right_ket), 2)
    env = deterministic_itensor((left, right_ket), 3)
    bra_bulk = deterministic_itensor((left, site, right_bra), 4)
    tmp = deterministic_itensor((right_ket, site, right_bra), 5)
    ket_bulk = deterministic_itensor((right_ket, site, next_ket), 6)

    return (;
        bra_first,
        ket_first,
        env,
        bra_bulk,
        tmp,
        ket_bulk,
    )
end

function run_case(fixtures, case::Symbol)
    if case == :first_site
        return dag(fixtures.bra_first) * fixtures.ket_first
    elseif case == :env_bra
        return fixtures.env * dag(fixtures.bra_bulk)
    elseif case == :tmp_ket
        return fixtures.tmp * fixtures.ket_bulk
    elseif case == :site_update
        tmp = fixtures.env * dag(fixtures.bra_bulk)
        return tmp * fixtures.ket_bulk
    else
        error("unknown case: $case")
    end
end

function run_for_seconds(f; warmup_seconds::Float64, measurement_seconds::Float64, min_samples::Int)
    sink = Ref{Any}(nothing)

    sink[] = f()
    GC.gc()

    warmup_start = time_ns()
    while (time_ns() - warmup_start) / 1.0e9 < warmup_seconds
        sink[] = f()
    end
    GC.gc()

    times_ns = Int[]
    measurement_start = time_ns()
    while (time_ns() - measurement_start) / 1.0e9 < measurement_seconds || length(times_ns) < min_samples
        start = time_ns()
        sink[] = f()
        push!(times_ns, time_ns() - start)
    end

    return sink[], times_ns
end

function print_result(; case, d, chi, value, times_ns)
    times_us = times_ns ./ 1.0e3
    @printf(
        "itensors_pairwise_%s,chi_%d_d_%d,%d,%.6f,%.6f,%.6f,%.6f,%s\n",
        String(case),
        chi,
        d,
        length(times_us),
        minimum(times_us),
        median(times_us),
        mean(times_us),
        maximum(times_us),
        string(inds(value)),
    )
end

function main()
    opts = parse_args(ARGS)
    BLAS.set_num_threads(opts["blas_threads"])

    d = opts["d"]
    chis = opts["chis"]
    warmup_seconds = opts["warm_up_time"]
    measurement_seconds = opts["measurement_time"]
    min_samples = opts["min_samples"]
    cases = [:first_site, :env_bra, :tmp_ket, :site_update]

    println("ITensors pairwise contraction benchmark")
    println("  Julia:            $(VERSION)")
    println("  ITensors:         $(Base.pkgversion(ITensors))")
    println("  threads:          Julia=$(Threads.nthreads()) BLAS=$(BLAS.get_num_threads())")
    println("  d:                $d")
    println("  chis:             $(join(chis, ","))")
    println("  warm-up time:     $warmup_seconds")
    println("  measurement time: $measurement_seconds")
    println("  min samples:      $min_samples")
    println()
    println("case,params,samples,min_us,median_us,mean_us,max_us,output_inds")

    for chi in chis
        fixtures = build_fixtures(d, chi)
        for case in cases
            value, times_ns = run_for_seconds(
                () -> run_case(fixtures, case);
                warmup_seconds,
                measurement_seconds,
                min_samples,
            )
            print_result(; case, d, chi, value, times_ns)
        end
    end
end

main()
