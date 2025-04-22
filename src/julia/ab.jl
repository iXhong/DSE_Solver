using FastGaussQuadrature
using NPZ


#Parameters
const D = (0.8^3)/0.5
const Nf = 4
const m0 = 0.0034
const ξ = 361
const γ_m = 12.0/25.0
const τ = exp(2)-1
const Λ_qcd = 0.234 # GeV
const mt = 0.5 # GeV
const ω = 0.5 # GeV


# Gluon propagator (Maris-Tandy & Qin-chang)
function G(k2::Float64)::Float64
    k2 = max(k2, 1e-10)
    part1 = 8.0 * π^2 / ω^4 * D * exp(-k2/ω^2)
    part2 = 8.0 * π^2 * γ_m / log(τ + (1.0 + k2 / Λ_qcd^2)^2)
    part3 = 4.0 * π^2 * D * exp(-k2/ω^2) / ω^4
    part4 = (1.0 - exp(-k2 / (4.0 * mt^2))) / k2
    # return part1 + part2 * F      #(Maris-Tandy)
    return part1 + part2 * part4        #Qin-chang

end


function gausslegendreGrid(xp1::Int32,xp2::Int32,Nz::Int32,Np::Int32)

    xz, wz = gausslegendre(Nz)
    xp, wp = gausslegendre(Np)
    xp1_log, xp2_log = log10(xp1), log10(xp2)
    xp = (xp2_log-xp1_log)*xp/2 .+ (xp1_log+xp2_log)/2
    wp = (xp2_log-xp1_log)/2 * wp
    xp = 10 .^xp
    wp = wp .*xp*log(10)

    return xz,wz,xp,wp
end


function solver(Nz::Int32,Np::Int32,xmin::Float64,xmax::Float64,max_iter::Int32,eps::Float64)
    local z2,z4
    reAi = ones(Float64,Np)
    reBi = fill(0.3,Np)
    reA = zeros(Np)
    reB = zeros(Np)
    xz, wz, xp, wp = gausslegendreGrid(xmin,xmax,Nz,Np)

    function intreAB(p2::Float64,fArn::Float64,fBrn::Float64)
        fA,fB = 0, 0
        for i = 1:Np
            for j = 1:Nz
                dxw = xp[i]*wp[i]*sqrt(1-xz[j]^2)*wz[j]/(xp(i)*reAi[i]^2+reBi[i]^2)
                pqz = sqrt(p2*xp[i])*xz[j]
                k2 = p2+xp[i]-2*pqz
                #integration
                fA += dxw*reAi[i]*G(k2)*(pqz+2*(p2-pqz)*(pqz-xp[i])/k2)
                fB += dxw*reBi[i]*G(k2)
            end
            
        end
        fA = 4*fA/(p2*3*8*pi^3)
        fB = 4*fB/(8*pi^3)
        Ax = 1 + fA - fArn
        Bx = m0 + fB - fBrn

        return Ax, Bx
    end
    
    for iter in 1:max_iter
        fArn = 1.0
        fBrn = m0
        error = 0

        fArn, fBrn = intreAB(ξ,fArn,fBrn)
        for i in 1:Np
            reA[i],reB[i] = intreAB(xp[i],fArn,fBrn) 
        end

        error = sum(abs.(reB .- reBi))
        reAi .= reA
        reBi .= reB

        println("Iteration=$(iter+1),error="$error)
        if error < eps
            z2 = 1 - fArn
            z4 = 1 - fBrn/m0
            println("Converged after $(iter+1) iterations")
            println("z2=$z2,z4="$z4)
            break
        end

    end

    return z2,z4,reA,reB,xp
end


function main()

    Nz=25
    Np=200
    p2_min,p2_max = 1e-4,1e4
    max_iter = 1000
    eps = 1e-3

    z2,z4,reA,reB,xp = solver(Nz,Np,p2_min,p2_max,max_iter,eps)
    M = reB ./ reA
    data_dict = Dict("A"=>reA,"B"=>reB,"M"=>M,"z2"=>z2,"z4"=>z4,"p2"=>xp)
    npzwrite("result.npz",data_dict)
end

main()