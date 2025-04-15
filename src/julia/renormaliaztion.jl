using QuadGK
using LinearAlgebra
using Plots
using CSV
using DataFrames

# Parameters
const D = 1.0
const Nf = 4
const m = 5.0 #GeV 
const ξ = 19.0 # GeV
const γ_m = 12.0 / (33.0 - 2.0 * Nf)
const τ = exp(2.0) - 1.0
const Λ_qcd = 0.234 # GeV
const mt = 0.5 # GeV
const ω = 0.4 # GeV

# Gluon propagator (Maris-Tandy)
function G(k2::Float64)::Float64
    k2 = max(k2, 1e-10)
    part1 = 4.0 * π^2 / ω^6 * D * k2 / ω^2
    part2 = 8.0 * π^2 * γ_m / log(τ + (1.0 + k2 / Λ_qcd^2)^2)
    F = (1.0 - exp(-k2 / (4.0 * mt^2))) / k2
    return part1 * part2 * F
end


# Angular integration for A
function IntegralOfAx(p2::Float64, q2::Float64)::Float64
    function integrand(x::Float64)::Float64
        k2 = p2 + q2 - 2.0 * sqrt(p2 * q2) * x
        k2 = max(k2, 1e-10)
        part1 = 2.0 * (p2 * q2 + x * (p2 + q2) * sqrt(p2 * q2) - p2 * q2 * x)
        part2 = sqrt(1.0 - x^2)
        return part2 * G(k2) / k2 * (sqrt(p2 * q2) * x + part1 / k2)
    end
    integral, _ = quadgk(integrand, -1.0, 1.0, rtol=1e-6)
    return integral
end


# Angular integration for B
function IntegralOfBx(p2::Float64, q2::Float64)::Float64
    function integrand(x::Float64)::Float64
        k2 = p2 + q2 - 2.0 * sqrt(p2 * q2) * x
        k2 = max(k2, 1e-10)
        part2 = sqrt(1.0 - x^2)
        return part2 * G(k2) / k2
    end
    integral, _ = quadgk(integrand, -1.0, 1.0, rtol=1e-6)
    return integral
end


# Momentum grid
const N = 100
const p2_min, p2_max = 1e-4, 1e4
const p2_grid = 10.0 .^ range(log10(p2_min), log10(p2_max), length=N)
const q2_grid = copy(p2_grid)
const dq2 = diff(q2_grid)


# Integrands
function IntegrandOfA(p2::Float64, q2::Float64, A_q2::Float64, B_q2::Float64, IntegralOfAx::Function)::Float64
    part1 = q2 * A_q2 / (q2 * A_q2^2 + B_q2^2)
    part2 = IntegralOfAx(p2, q2)
    return part1 * part2
end

function IntegrandOfB(p2::Float64, q2::Float64, A_q2::Float64, B_q2::Float64, IntegralOfBx::Function)::Float64
    part1 = q2 * B_q2 / (q2 * A_q2^2 + B_q2^2)
    part2 = IntegralOfBx(p2, q2)
    return part1 * part2
end


function compute_Integrand(p2::Float64,A::Vector{Float64},B::Vector{Float64})
    IA = 0.0
    IB = 0.0
    for j in 1:N-1
        q2 = q2_grid[j]
        IA += IntegrandOfA(p2, q2, A[j], B[j], IntegralOfAx) * dq2[j]
        IB += IntegrandOfB(p2, q2, A[j], B[j], IntegralOfBx) * dq2[j]
    end
    return IA, IB
end


# Iterative solver
function iterate_AB(Z2::Float64, Z4::Float64, A::Vector{Float64}, B::Vector{Float64}; tol::Float64=10.0, max_iter::Int=10)
    A_new = zeros(Float64, N)
    B_new = zeros(Float64, N)
    A_prime = zeros(Float64, N)
    B_prime = zeros(Float64, N)

    for iteration in 1:max_iter
        Threads.@threads for i in 1:N
            p2 = p2_grid[i]
            IA,IB = compute_Integrand(p2,A,B)
            A_prime[i] = 1.0 / (6.0 * p2 * π^3) * IA
            A_new[i] = Z2 + A_prime[i]
            B_prime[i] = 1.0 / (2.0 * π^2) * IB
            B_new[i] = Z4 * m + B_prime[i]
        end
        delta_A = maximum(abs.(A_new .- A))
        delta_B = maximum(abs.(B_new .- B))
        println("Iteration $iteration: max|A_new - A| = $delta_A, max|B_new - B| = $delta_B")
        if delta_A < tol && delta_B < tol
            println("Converged in $iteration iterations")
            break
        end
        A .= A_new
        B .= B_new
    end
    return A, B, A_prime, B_prime
end


# Compute Z2 and Z4
function findZ2Z4(A::Vector{Float64}, B::Vector{Float64})
    Z2_init = 1.0
    Z4_init = 1.0
    xi_index = argmin(abs.(p2_grid .- ξ))
    A, B, A_prime, B_prime = iterate_AB(Z2_init, Z4_init, A, B)
    Z2 = 1.0 - A_prime[xi_index]
    Z4 = 1.0 - B_prime[xi_index] / m
    println("A(ξ) = $(A[xi_index]), expected ≈ 1")
    println("B(ξ) = $(B[xi_index]), expected ≈ $m")
    return Z2, Z4, A, B
end

#save results to csv 
function savecsv(A,B,M,Z2,Z4)
    data = DataFrame(
        A = A,
        B = B,
        M = M,
        Z2 = Z2,
        Z4 = Z4
    )

    CSV.write("../../results/data/julia/data.csv",data)
    
end


function dataPlot(A,B,M,p2_grid)
        # Plot A, B, and M
        p1 = plot(p2_grid, A, xscale=:log10, label="A(p²)", xlabel="p² (GeV²)", ylabel="A", title="Self-Energy")
        plot!(p1, p2_grid, B, label="B(p²)")
        p2 = plot(p2_grid, M, xscale=:log10, label="M(p²)", xlabel="p² (GeV²)", ylabel="M (GeV)", title="Effective Mass")
        plot(p1, p2, layout=(2,1))
        savefig("../../results/test.png")
    
end


# Main function
function main()
    A = ones(Float64, N)
    B = zeros(Float64, N) # Try B = fill(0.3, N) for light quarks
    Z2, Z4, A, B = findZ2Z4(A, B)
    println("Z2 = $Z2, Z4 = $Z4")

    # Compute effective mass
    M = B ./ A
    # println("M(0) = $(M[1]) GeV")

    # data_dict = Dict("A"=>A,"B"=>B,"M"=>M,"Z2"=>Z2,"Z4"=>Z4)
    # npzwrite("./data/result.npz",data_dict)

    savecsv(A,B,M,Z2,Z4)
    dataPlot(A,B,M,p2_grid)

end

main()