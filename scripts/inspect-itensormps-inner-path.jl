#!/usr/bin/env julia

using LinearAlgebra
using ITensors
using ITensorMPS

function parse_args(args)
    opts = Dict{String, Int}("L" => 4, "d" => 2, "chi" => 4)
    i = 1
    while i <= length(args)
        if args[i] == "--L"
            i += 1
            opts["L"] = parse(Int, args[i])
        elseif args[i] == "--d"
            i += 1
            opts["d"] = parse(Int, args[i])
        elseif args[i] == "--chi"
            i += 1
            opts["chi"] = parse(Int, args[i])
        else
            error("unknown argument: $(args[i])")
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

function deterministic_mps(sites::Vector, chi::Int; prefix::String, seed_offset::Int)::MPS
    nsites = length(sites)
    links = [Index(chi, "Link,$prefix,l=$n") for n in 1:(nsites - 1)]
    tensors = Vector{ITensor}(undef, nsites)
    for site in 1:nsites
        inds_tuple =
            if nsites == 1
                (sites[site],)
            elseif site == 1
                (sites[site], links[site])
            elseif site == nsites
                (links[site - 1], sites[site])
            else
                (links[site - 1], sites[site], links[site])
            end
        tensors[site] = deterministic_itensor(inds_tuple, seed_offset + site)
    end
    return MPS(tensors)
end

function fmt_index(i)
    return "$(i) dim=$(dim(i))"
end

function fmt_inds(T::ITensor)
    return "[" * join(fmt_index.(collect(inds(T))), " | ") * "]"
end

function fmt_common(A::ITensor, B::ITensor)
    c = collect(commoninds(A, B))
    return "[" * join(fmt_index.(c), " | ") * "]"
end

function main()
    opts = parse_args(ARGS)
    BLAS.set_num_threads(1)
    L = opts["L"]
    d = opts["d"]
    chi = opts["chi"]
    sites = [Index(d, "Site,n=$site") for site in 1:L]
    bra = deterministic_mps(sites, chi; prefix = "bra", seed_offset = 0)
    ket = deterministic_mps(sites, chi; prefix = "ket", seed_offset = L)

    bra_dag = dag(bra)
    sim!(linkinds, bra_dag)

    println("ITensorMPS inner path")
    println("L=$L d=$d chi=$chi")
    println("Initial bra_dag/ket site index compatibility:")
    for site in 1:L
        println("site $site")
        println("  bra_dag[$site] inds = $(fmt_inds(bra_dag[site]))")
        println("  ket[$site]     inds = $(fmt_inds(ket[site]))")
        println("  common         = $(fmt_common(bra_dag[site], ket[site]))")
    end

    println()
    println("Contraction trace:")
    O = bra_dag[1] * ket[1]
    println("site 1: O = bra_dag[1] * ket[1]")
    println("  contracted common = $(fmt_common(bra_dag[1], ket[1]))")
    println("  O inds            = $(fmt_inds(O))")

    for site in 2:L
        println("site $site: first half P = O * bra_dag[$site]")
        println("  O inds            = $(fmt_inds(O))")
        println("  bra_dag[$site] inds = $(fmt_inds(bra_dag[site]))")
        println("  contracted common = $(fmt_common(O, bra_dag[site]))")
        P = O * bra_dag[site]
        println("  P inds            = $(fmt_inds(P))")

        println("site $site: second half O = P * ket[$site]")
        println("  ket[$site] inds     = $(fmt_inds(ket[site]))")
        println("  contracted common = $(fmt_common(P, ket[site]))")
        O = P * ket[site]
        println("  O inds            = $(fmt_inds(O))")
    end

    println()
    println("final scalar = $(O[])")
end

main()
