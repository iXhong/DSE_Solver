using QuadGK
using LinearAlgebra

# Define parameters
const D = 1.0
const Nf = 4
const m = 0.005
const xi = 19.0 # GeV
const gamma_m = 12.0 / (33.0 - 2.0 * Nf)
const tau = exp(2.0) - 1.0
const lambda_qcd = 0.234 # GeV
const mt = 0.5 # GeV
const omega = 0.4 # GeV

# Define Maris-Tandy gluon propagator G(k²)
function G(k2::Float64)::Float64
    k2 = max(k2, 1e-10)
    part1 = 4.0 * π^2 / omega^6 * D * k2 / omega^2
    part2 = 8.0 * π^2 * gamma_m / log(tau + (1.0 + k2 / lambda_qcd^2)^2)
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
    integral, _ = quadgk(integrand, -1.0, 1.0, rtol=1e-8)
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
    integral, _ = quadgk(integrand, -1.0, 1.0, rtol=1e-8)
    return integral
end

# Momentum grid
const N = 100
const p2_min, p2_max = 1e-4, 1e3 # p² range
const p2_grid = 10.0 .^ range(log10(p2_min), log10(p2_max), length=N)
const q2_grid = copy(p2_grid)
const dq2 = diff(q2_grid)

# Integrand for A
function IntegrandOfA(p2::Float64, q2::Float64, A_q2::Float64, B_q2::Float64, IntegralOfAx::Function)::Float64
    part1 = q2 * A_q2 / (q2 * A_q2^2 + B_q2^2)
    part2 = IntegralOfAx(p2, q2)
    return part1 * part2
end

# Integrand for B
function IntegrandOfB(p2::Float64, q2::Float64, A_q2::Float64, B_q2::Float64, IntegralOfBx::Function)::Float64
    part1 = q2 * B_q2 / (q2 * A_q2^2 + B_q2^2)
    part2 = IntegralOfBx(p2, q2)
    return part1 * part2
end

# Iterative solver for A and B
function iterate_AB(Z2::Float64, Z4::Float64, A::Vector{Float64}, B::Vector{Float64}; tol::Float64=1e-6, max_iter::Int=100)
    A_new = zeros(Float64, N)
    B_new = zeros(Float64, N)
    A_prime = zeros(Float64, N)
    B_prime = zeros(Float64, N)

    for iteration in 1:max_iter
        for i in 1:N
            p2 = p2_grid[i]
            IA = 0.0
            IB = 0.0

            for j in 1:N-1
                q2 = q2_grid[j]
                IA += IntegrandOfA(p2, q2, A[j], B[j], IntegralOfAx) * dq2[j]
                IB += IntegrandOfB(p2, q2, A[j], B[j], IntegralOfBx) * dq2[j]
            end

            A_prime[i] = 1.0 / (6.0 * p2 * π^3) * IA
            A_new[i] = Z2 + A_prime[i]

            B_prime[i] = 1.0 / (2.0 * π^2) * IB
            B_new[i] = Z4 * m + B_prime[i]
        end

        if maximum(abs.(A_new .- A)) < tol && maximum(abs.(B_new .- B)) < tol
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
    xi_index = argmin(abs.(p2_grid .- xi))

    A, B, A_prime, B_prime = iterate_AB(Z2_init, Z4_init, A, B)

    Z2 = 1.0 - A_prime[xi_index]
    Z4 = 1.0 - B_prime[xi_index] / m

    return Z2, Z4, A, B
end

# Main execution
function main()
    # Initialize A and B
    A = ones(Float64, N)
    B = zeros(Float64, N)

    # Compute Z2 and Z4
    Z2, Z4, A, B = findZ2Z4(A, B)

    println("Z2 = $Z2, Z4 = $Z4")

    # Optionally return A, B for further analysis
    return Z2, Z4, A, B
end

# Run the code
Z2, Z4, A, B = main()
