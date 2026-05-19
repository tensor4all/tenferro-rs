#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Statistics
using ITensors
using ITensorMPS

const DEFAULT_L = 32
const DEFAULT_D = 2
const DEFAULT_CHIS = [4, 8, 16, 32, 64]
const DEFAULT_WARMUP_SECONDS = 1.0
const DEFAULT_MEASUREMENT_SECONDS = 2.0
const DEFAULT_MIN_SAMPLES = 10

function usage()
    println("""
Usage: julia --project=<env-with-ITensors-ITensorMPS> scripts/bench-itensormps-inner-product.jl [options]

Options:
  --L N                         MPS length (default: $(DEFAULT_L))
  --d N                         Physical dimension (default: $(DEFAULT_D))
  --chis LIST                   Comma-separated bond dimensions (default: $(join(DEFAULT_CHIS, ",")))
  --warm-up-time SECONDS        Warm-up time after the first JIT call (default: $(DEFAULT_WARMUP_SECONDS))
  --measurement-time SECONDS    Measurement time per chi (default: $(DEFAULT_MEASUREMENT_SECONDS))
  --min-samples N               Minimum samples per chi (default: $(DEFAULT_MIN_SAMPLES))
  --blas-threads N              Julia BLAS threads (default: 1)
  --help                        Show this help text
""")
end

function parse_args(args)
    opts = Dict{String, Any}(
        "L" => DEFAULT_L,
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
        elseif arg == "--L"
            i += 1
            opts["L"] = parse(Int, args[i])
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

function deterministic_itensor(inds::Tuple, seed::Int)::ITensor
    dims = map(dim, inds)
    len = prod(dims)
    data = Vector{ComplexF64}(undef, len)
    @inbounds for pos in eachindex(data)
        data[pos] = complex_value(pos - 1, seed)
    end
    return ITensor(reshape(data, dims...), inds...)
end

function deterministic_mps(
    sites::Vector,
    chi::Int;
    prefix::String,
    seed_offset::Int,
)::MPS
    nsites = length(sites)
    links = [Index(chi, "Link,$prefix,l=$n") for n in 1:(nsites - 1)]
    tensors = Vector{ITensor}(undef, nsites)

    for site in 1:nsites
        inds =
            if nsites == 1
                (sites[site],)
            elseif site == 1
                (sites[site], links[site])
            elseif site == nsites
                (links[site - 1], sites[site])
            else
                (links[site - 1], sites[site], links[site])
            end
        tensors[site] = deterministic_itensor(inds, seed_offset + site)
    end

    return MPS(tensors)
end

function run_for_seconds(f; warmup_seconds::Float64, measurement_seconds::Float64, min_samples::Int)
    sink = Ref{Any}(nothing)

    sink[] = f() # Compile and populate any method caches before timed warm-up.
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

function print_result(; L, d, chi, value, times_ns)
    times_ms = times_ns ./ 1.0e6
    @printf(
        "itensor_mps_inner,L_%d_chi_%d_d_%d,%d,%.6f,%.6f,%.6f,%.6f,%s\n",
        L,
        chi,
        d,
        length(times_ms),
        minimum(times_ms),
        median(times_ms),
        mean(times_ms),
        maximum(times_ms),
        repr(value),
    )
end

function main()
    opts = parse_args(ARGS)
    BLAS.set_num_threads(opts["blas_threads"])

    L = opts["L"]
    d = opts["d"]
    chis = opts["chis"]
    warmup_seconds = opts["warm_up_time"]
    measurement_seconds = opts["measurement_time"]
    min_samples = opts["min_samples"]

    println("ITensorMPS MPS inner benchmark")
    println("  Julia:            $(VERSION)")
    println("  ITensors:         $(Base.pkgversion(ITensors))")
    println("  ITensorMPS:       $(Base.pkgversion(ITensorMPS))")
    println("  threads:          Julia=$(Threads.nthreads()) BLAS=$(BLAS.get_num_threads())")
    println("  L:                $L")
    println("  d:                $d")
    println("  chis:             $(join(chis, ","))")
    println("  warm-up time:     $warmup_seconds")
    println("  measurement time: $measurement_seconds")
    println("  min samples:      $min_samples")
    println()
    println("case,params,samples,min_ms,median_ms,mean_ms,max_ms,value")

    for chi in chis
        sites = [Index(d, "Site,n=$site") for site in 1:L]
        bra = deterministic_mps(sites, chi; prefix = "bra", seed_offset = 0)
        ket = deterministic_mps(sites, chi; prefix = "ket", seed_offset = L)

        value, times_ns = run_for_seconds(
            () -> inner(bra, ket);
            warmup_seconds,
            measurement_seconds,
            min_samples,
        )
        print_result(; L, d, chi, value, times_ns)
    end
end

main()
