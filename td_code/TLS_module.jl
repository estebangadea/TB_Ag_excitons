using SparseArrays
include("IOmodule.jl")

###Constants###
const hbar=1
const e_m=1 #electron mass
const e_c=1 #electron charge
const fsc=1/137 #alpha, fine-structure constant
const sol=137 #speed of light
const Kb = 8.6173324e-5/27.2113862


#FUNCTIONS#
function construct_aob_hamiltonian(nc::Int64, hop1::Float64, hop2::Float64)
    hrows = Int64[]
    hcols = Int64[]
    hvals = ComplexF64[]

    for ii = 1:2:2*nc-1
        push!(hvals, hop1)
        push!(hcols, ii+1)
        push!(hrows, ii)
        push!(hvals, hop1)
        push!(hcols, ii)
        push!(hrows, ii+1)
    end
    
    for ii = 2:2:2*nc-1
        push!(hvals, hop2)
        push!(hcols, ii+1)
        push!(hrows, ii)
        push!(hvals, hop2)
        push!(hcols, ii)
        push!(hrows, ii+1)
    end

    push!(hvals, hop2)
    push!(hcols, 1)
    push!(hrows, 2*nc)
    push!(hvals, hop2)
    push!(hcols, 2*nc)
    push!(hrows, 1)

    Haob = sparse(hrows, hcols, hvals, 2*nc, 2*nc)

    return Haob
end

function construct_gs_dens(solv::Array{Complex{Float64},2}, isolv::Array{Complex{Float64},2}, dimer::Float64)
    nc = size(solv)[1]
    Rho = zeros(ComplexF64,nc,nc)
    for ii = 1:Int(nc/2)
        Rho[ii,ii] = 2
    end
    if dimer < 1e-8
        Rho[Int(nc/2),Int(nc/2)] = 1
        Rho[Int(nc/2)+1,Int(nc/2)+1] = 1
    end
    Rho_aob = solv*Rho*isolv
    return Rho_aob
end


function commutator!(
    alpha::Number, A::SparseMatrixCSC, B::AbstractMatrix,
    beta::Number, C::AbstractMatrix)
if beta != 1
    C .*= beta
end
for j = 1: A.n
    for k = 1:A.n, i_val = A.colptr[k] : (A.colptr[k+1]-1)
        i = A.rowval[i_val]
        A_ik = A.nzval[i_val]
        C[i,j] = C[i,j] + alpha * A_ik * B[k,j]
    end
end
for j = 1:A.n, i_val = A.colptr[j] : (A.colptr[j+1]-1)
    k = A.rowval[i_val]
    A_kj = A.nzval[i_val]
    for i = 1: A.n
        C[i,j] = C[i,j] - alpha * B[i, k] * A_kj
    end
end
end

function RungeCuta(dens::Array{Complex{Float64},2},Hn::SparseMatrixCSC{Complex{Float64}},tstep::Float64)
    k1 = copy(dens)
    commutator!(-1im*0.5*tstep, Hn, dens, 1, k1)# k1 = dens * 1 + (-1im * 0.5 * tstep) * [Hn, dens]
    commutator!(-1im*tstep, Hn, k1, 1, dens) # 
end

function VPstep(stop::Float64, height::Float64, stepi::Int64, timestep::Float64)
    if stepi*timestep < stop
        A = height/stop * (stop/(2*pi) * sin(stepi*timestep*2*pi/stop+pi)+stepi*timestep)
    else
        A = height
    end
    return A
end

function VPgaus(stop::Float64, height::Float64, stepi::Int64, timestep::Float64)  
    if stepi*timestep < stop
        A = height * exp(-1 * (stepi*timestep - stop/2.0)^2 / (2*(stop/10)^2))
    else
        A = height*stop
    end
    return A
end

function VPlaser(stop::Float64, height::Float64, freq::Float64, stepi::Int64, timestep::Float64)
    ti = stepi*timestep
    if ti < stop
        k1 = -2 / freq * cos(stop * freq / -2 + freq * ti)
        k2 = stop / (-2 * pi + stop * freq) * cos(stop * freq / -2 - 2 * pi * ti / stop + freq * ti)
        k3 = stop / (2 * pi + stop * freq) * cos(stop * freq / -2 + 2 * pi * ti / stop + freq * ti)
        k4 = 4 * pi^2 * cos(stop * freq / 2) / (stop^2 * freq^3 - 4 * pi^2 * freq) 
        A = height * (0.5 * (k1 + k2 + k3) - k4)  +  ti * 0.0005
    else
        A = ti * 0.0005   #0.0
    end
    return A
end

function VPconst(tau::Float64, A0::Float64, stepi::Int64, timestep::Float64)  
    ti = stepi*timestep
    Ft = A0 / tau * ( tau^2 / (4 * pi^2) * (cos(2 * pi * ti / tau) - 1) + ti^2 / 2)
    if ti < tau
        A = Ft
    else
        A =  A0 * ti - A0 * tau / 2
    end
    return A
end

function VectorPotMat(Hi::SparseMatrixCSC{Complex{Float64}}, H0::SparseMatrixCSC{Complex{Float64}}, vectpot::Float64, size::Int64, dip1::Number, dip2::Number)
    beta1 = exp(1im * vectpot * dip1)
    beta2 = exp(1im * vectpot * dip2)
    for ii = 1:2:size-1
        Hi[ii,ii+1] = H0[ii,ii+1] * beta1
        Hi[ii+1,ii] = H0[ii+1,ii] * conj(beta1)
    end
    for ii = 2:2:size-1
        Hi[ii,ii+1] = H0[ii,ii+1] * beta2
        Hi[ii+1,ii] = H0[ii+1,ii] * conj(beta2)
    end
    Hi[1,size] = H0[1,size] * conj(beta2)
    Hi[size,1] = H0[size,1] * beta2
end

function VectorPotMat(Hi::SparseMatrixCSC{Complex{Float64}}, H0::SparseMatrixCSC{Complex{Float64}}, vectpot::Float64, size::Int64, rion::Array{Float64,1}, boxl::Float64)
    for ii = 1:size-1
        Hi[ii,ii+1] = Hi[ii,ii+1] * exp(1im * vectpot * (rion[ii+1]-rion[ii]))
        Hi[ii+1,ii] = Hi[ii+1,ii] * exp(-1im * vectpot * (rion[ii+1]-rion[ii]))
    end
    Hi[1,size] = Hi[1,size] * exp(-1im * vectpot * (rion[1]+boxl-rion[size]))
    Hi[size,1] = Hi[size,1] * exp(1im * vectpot * (rion[1]+boxl-rion[size]))
end

function Hxc(Hcd::SparseMatrixCSC{Complex{Float64}}, Rho::Array{Complex{Float64},2}, q0::Array{Complex{Float64},1}, size::Int64, xc::Array{Complex{Float64},2})
    
    dq = (diag(Rho) .- q0)' * xc
    for jj in 1:size
        Hcd[jj,jj] = Hcd[jj,jj] + dq[jj]
    end
end

function Honsite(Hcd::SparseMatrixCSC{Complex{Float64}}, Rho::Array{Complex{Float64},2}, q0::Array{Complex{Float64},1}, size::Int64, U::Float64)
    # Pure on-site (Hubbard-like) feedback: U*(rho_jj - q0_jj), no coupling to other sites at
    # all -- unlike Hxc's hartreeu, which has a 1/U spatial softening radius and so still
    # couples to neighbors. This term has no dependence on ion positions, so it contributes
    # exactly zero to the Hellmann-Feynman ionic force (no change needed in build_fion).
    for jj in 1:size
        Hcd[jj,jj] = Hcd[jj,jj] + U*(Rho[jj,jj]-q0[jj])
    end
end

function Hint(Hcd1::SparseMatrixCSC{Complex{Float64}}, Hcd2::SparseMatrixCSC{Complex{Float64}}, Rho1::Array{Complex{Float64},2}, Rho2::Array{Complex{Float64},2}, q01::Array{Complex{Float64},1}, q02::Array{Complex{Float64},1}, size1::Int64, size2::Int64, intch::Array{Float64,2})
    
    dq1 = (diag(Rho2) .- q02)' * intch'
    dq2 = (diag(Rho1) .- q01)' * intch

    for jj in 1:size1
        Hcd1[jj,jj] = Hcd1[jj,jj] + dq1[jj]
    end
    for jj in 1:size2
        Hcd2[jj,jj] = Hcd2[jj,jj] + dq2[jj]
    end
end

function buildxc(rion::Array{Float64,1}, boxl::Float64, alpha::Number, gamma::Number, U::Float64, size::Int64)
    # Hartree term: 1/sqrt(d^2 + 1/U^2), strength fixed (K=1), softened so its on-site (d=0) value is exactly U.
    # fxc term:     alpha/sqrt(d^2 + gamma^2). Both run over ALL site pairs, including the on-site (d=0) self term,
    # matching report eqs 6-7 (no separate on-site/off-site cases).
    # array is symmetric and array2 antisymmetric in (ii,jj) (dij flips sign, dij^2 doesn't), so only the
    # ii<=jj triangle is computed and mirrored -- halves the O(size^2) cost of this hot per-step routine.
    array = zeros(ComplexF64, size, size)
    array2 = zeros(ComplexF64, size, size)
    Usqd = U^2
    gammasqd = gamma^2
    @inbounds for jj = 1:size
        for ii = 1:jj
            dij = (rion[jj]-rion[ii]) - ((rion[jj]-rion[ii]) ÷ (boxl/2)) * boxl
            s1 = dij^2 + 1/Usqd
            s2 = dij^2 + gammasqd
            sq1 = sqrt(s1)
            sq2 = sqrt(s2)
            a  = 1/sq1 + alpha/sq2
            a2 = -dij/(s1*sq1) - alpha*dij/(s2*sq2)
            array[ii,jj] = a
            array[jj,ii] = a
            array2[ii,jj] = a2
            array2[jj,ii] = -a2
        end
    end
    return array, array2
end

function buildintchain(rion1::Array{Float64,1}, rion2::Array{Float64,1}, boxl::Float64, input::Input)
    ysqd = input.rchain^2

    size1 = 2*input.nchain1
    size2 = 2*input.nchain2

    array = zeros(Float64, size1, size2)
    array2 = zeros(Float64, size1, size2)

    @inbounds for jj = 1:size2
        for ii = 1:size1
            dij = (rion1[ii]-rion2[jj]) - ((rion1[ii]-rion2[jj]) ÷ (boxl/2)) * boxl
            s = dij^2 + ysqd
            sq = sqrt(s)
            array[ii,jj] = 1/sq
            array2[ii,jj] = -dij/(s*sq)
        end
    end
    return array, array2
end

function construct_rdep_hamiltonian(focki::SparseMatrixCSC{Complex{Float64}}, fockaob::SparseMatrixCSC{Complex{Float64}}, rion::Array{Float64,1}, boxl, pottable::Array{Float64,2}, input::Input, size::Int64)
    for ii in 1:size
        focki[ii,ii] = fockaob[ii,ii]
    end
    for ii = 1:size-1
        index = Int64(round( ( (rion[ii+1]-rion[ii]) /input.req - 0.5) * 10000/1.5 ) )
        focki[ii,ii+1] = pottable[index, 2]
        focki[ii+1,ii] = pottable[index, 2]
    end
    index = Int64(round( ( (rion[1]+boxl-rion[size]) /input.req - 0.5) * 10000/1.5 ) )
    focki[size, 1] = pottable[index, 2]
    focki[1, size] = pottable[index, 2]
end

function build_fion(input::Input, rion::Array{Float64,1}, vion::Array{Float64,1}, boxl::Float64, pottable::Array{Float64,2}, rhoi::Array{Complex{Float64},2}, xc::Array{Complex{Float64},2}, dint::Array{Float64,1})
    size = length(rion)
    fion = zeros(Float64, size)
    for ii = 1:size-1
        index = Int64(round( ( (rion[ii+1]-rion[ii]) /input.req - 0.5) * 10000/1.5 ) )
        rhoint = 2 * real.(rhoi[ii,ii+1])
        fion[ii]    += pottable[index, 4] * rhoint + pottable[index, 5]
        fion[ii+1]  -= pottable[index, 4] * rhoint + pottable[index, 5]
    end
    index = Int64(round( ( (rion[1]+boxl-rion[size]) /input.req - 0.5) * 10000/1.5 ) )
    rhoint = 2 * real.(rhoi[1,size])
    fion[size] += pottable[index, 4] * rhoint + pottable[index, 5]
    fion[1] -= pottable[index, 4] * rhoint + pottable[index, 5]

    if input.fxcalpha^2 + input.hartreeu^2 > 10^-9
        fxcion = ((real(diag(rhoi)) .- 1.0)' * xc)
        for jj = 1:size
            fion[jj] += real(rhoi[jj,jj]-1.0) * fxcion[jj]
        end
    end

    if input.damp == 1
        for jj = 1:size
            fion[jj] -= input.dampforce * vion[jj]
        end
    end

    if input.chains == 2
        for jj = 1:size
            fion[jj] += dint[jj]
        end
    end

    return fion
end

function build_dint(rho1i::Array{Complex{Float64},2}, rho2i::Array{Complex{Float64},2}, gspop1::Array{Complex{Float64},1}, gspop2::Array{Complex{Float64},1}, dintchm::Array{Float64,2})
    size1 = length(gspop1)
    size2 = length(gspop2)
    array1 = zeros(Float64, size1)
    array2 = zeros(Float64, size2)

    intpot1 = (dintchm * (real(diag(rho2i)) .- 1.0))'
    intpot2 = ((real(diag(rho1i)) .- 1.0)' * dintchm)

    for jj = 1:size1
        array1[jj] += real(rho1i[jj,jj]-1.0) * intpot1[jj]
    end
    for jj = 1:size2
        array2[jj] += real(rho2i[jj,jj]-1.0) * intpot2[jj]
    end

    return array1, array2
end

function ionenergy(rion::Array{Float64,1}, vion::Array{Float64,1}, atmass::Float64, boxl::Float64, pottable::Array{Float64,2})
    energy = 0.0
    size = length(rion)
    for ii = 1:size-1
        index = Int64(round( ( (rion[ii+1]-rion[ii]) /input.req - 0.5) * 10000/1.5 ) )
        energy    += pottable[index, 3]
    end
    index = Int64(round( ( (rion[1]+boxl-rion[size]) /input.req - 0.5) * 10000/1.5 ) )
    energy += pottable[index, 3]
    energy += sum(atmass / 2 * vion.^2)
    return energy
end

function velverlet(rion::Array{Float64,1}, fion::Array{Float64,1}, vion::Array{Float64,1}, input::Input, boxl::Float64, pottable::Array{Float64,2}, rhoi::Array{Complex{Float64},2}; dlrxc::Array{Complex{Float64},2} = zeros(ComplexF64,2,2), dint::Array{Float64,1} = zeros(Float64,2))
    nstep = input.tstep/2
    rion .= rion .+ (vion .* nstep) .+ (0.5/input.atmass * nstep^2 .* fion)
    vion .= vion .+ 0.5/input.atmass * nstep .* fion
    fion .= build_fion(input, rion, vion, boxl, pottable, rhoi, dlrxc, dint)
    vion .= vion .+ 0.5/input.atmass * nstep .* fion
end


function propagate(rhoi::Array{Complex{Float64},2}, gspop::Array{Complex{Float64},1},
    focki::SparseMatrixCSC{Complex{Float64}}, fockaob::SparseMatrixCSC{Complex{Float64}},
    rion::Array{Float64,1}, vion::Array{Float64,1}, fion::Array{Float64,1},
    boxl::Float64, pottable::Array{Float64,2}, input::Input, ii::Int64;
    xc_cache = nothing) # when ehrenfest=0, rion never changes, so the caller can pass a
    # precomputed (lrcxco, dlrxco) here instead of paying buildxc's O(size^2) cost on every step

    construct_rdep_hamiltonian(focki, fockaob, rion, boxl, pottable, input, 2*input.nchain1) # Construct H based on position

    if ii < input.vtime/input.tstep #apply vector potentials
        if input.vpot == 1
            A = VPstep(input.vtime, input.vamp, ii, input.tstep)
            VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
        elseif input.vpot == 2
            A = VPgaus(input.vtime, input.vamp, ii, input.tstep)
            VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
        elseif input.vpot == 3
            A = VPlaser(input.vtime, input.vamp, input.vfreq, ii, input.tstep)
            VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
        elseif input.vpot == 4
            A = VPconst(input.vtime, input.vamp, ii, input.tstep)
            VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
        end
    elseif input.vpot == 1
        A = input.vamp
        VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
    elseif input.vpot == 3
        A = VPlaser(input.vtime, input.vamp, input.vfreq, ii, input.tstep)
        VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
    elseif input.vpot == 4
        A = VPconst(input.vtime, input.vamp, ii, input.tstep)
        VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
    end

    if input.fxcalpha^2 + input.hartreeu^2 > 1e-9
        lrcxco, dlrxco = xc_cache === nothing ? buildxc(rion, boxl, input.fxcalpha, input.fxcgamma, input.hartreeu, 2*input.nchain1) : xc_cache
        Hxc(focki, rhoi, gspop, 2*input.nchain1, lrcxco)
        if input.hubbardu^2 > 1e-9
            Honsite(focki, rhoi, gspop, 2*input.nchain1, input.hubbardu)
        end
        if input.ehrenfest == 1
            velverlet(rion, fion, vion, input, boxl, pottable, rhoi, dlrxc=dlrxco)
        end
    else
        if input.hubbardu^2 > 1e-9
            Honsite(focki, rhoi, gspop, 2*input.nchain1, input.hubbardu)
        end
        if input.ehrenfest == 1
            velverlet(rion, fion, vion, input, boxl, pottable, rhoi)
        end
    end
    if input.propflag == 0
        RungeCuta(rhoi,focki,input.tstep)
    else
        densint = copy(rhoi)
        commutator!(-1im*0.5*input.tstep, focki, rhoi, 1, densint)
        construct_rdep_hamiltonian(focki, fockaob, rion, boxl, pottable, input, 2*input.nchain1)

        if ii < input.vtime/input.tstep
            if input.vpot == 1
                A = VPstep(input.vtime, input.vamp, ii, input.tstep)
                VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
            elseif input.vpot == 2
                A = VPgaus(input.vtime, input.vamp, ii, input.tstep)
                VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
            elseif input.vpot == 3
                A = VPlaser(input.vtime, input.vamp, input.vfreq, ii, input.tstep)
                VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
            elseif input.vpot == 4
                A = VPconst(input.vtime, input.vamp, ii, input.tstep)
                VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
            end
        elseif input.vpot == 1
            A = input.vamp
            VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
        elseif input.vpot == 3
            A = VPlaser(input.vtime, input.vamp, input.vfreq, ii, input.tstep)
            VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
        elseif input.vpot == 4
            A = VPconst(input.vtime, input.vamp, ii, input.tstep)
            VectorPotMat(focki, fockaob, A, 2*input.nchain1, rion, boxl)
        end

        if input.fxcalpha^2 + input.hartreeu^2 > 1e-9
            lrcxco, dlrxco = xc_cache === nothing ? buildxc(rion, boxl, input.fxcalpha, input.fxcgamma, input.hartreeu, 2*input.nchain1) : xc_cache
            Hxc(focki, densint, gspop, 2*input.nchain1, lrcxco)
        end
        if input.hubbardu^2 > 1e-9
            Honsite(focki, densint, gspop, 2*input.nchain1, input.hubbardu)
        end

        if input.propflag == 1
            commutator!(-1im*input.tstep, focki, densint, 1, rhoi)
        elseif input.propflag == 2
            rhoi .= exp(-1im * collect(focki) * input.tstep) * rhoi * exp(1im * collect(focki) * input.tstep)
        end
    end
end

function propagate(rho1i::Array{Complex{Float64},2}, rho2i::Array{Complex{Float64},2},
    gspop1::Array{Complex{Float64},1}, gspop2::Array{Complex{Float64},1},
    fock1i::SparseMatrixCSC{Complex{Float64}}, fock2i::SparseMatrixCSC{Complex{Float64}},
    fock1aob::SparseMatrixCSC{Complex{Float64}}, fock2aob::SparseMatrixCSC{Complex{Float64}},
    rion1::Array{Float64,1}, rion2::Array{Float64,1},
    vion1::Array{Float64,1}, vion2::Array{Float64,1},
    fion1::Array{Float64,1}, fion2::Array{Float64,1},
    boxl::Float64, pottable::Array{Float64,2}, input::Input, ii::Int64;
    xc_cache = nothing) # when ehrenfest=0, rion1/rion2 never change, so the caller can pass a
    # precomputed (lrcxco1, dlrxco1, lrcxco2, dlrxco2, intchm, dintchm) here instead of paying
    # buildxc/buildintchain's O(size^2) cost on every step

    construct_rdep_hamiltonian(fock1i, fock1aob, rion1, boxl, pottable, input, 2*input.nchain1) # Construct H based on position
    construct_rdep_hamiltonian(fock2i, fock2aob, rion2, boxl, pottable, input, 2*input.nchain2) # Construct H based on position
    intchm, dintchm = xc_cache === nothing ? buildintchain(rion1, rion2, boxl, input) : (xc_cache[5], xc_cache[6])
    Hint(fock1i, fock2i, rho1i, rho2i, gspop1, gspop2, 2*input.nchain1, 2*input.nchain2, intchm)
    dint1, dint2 = build_dint(rho1i, rho2i, gspop1, gspop2, dintchm)

    if ii < input.vtime/input.tstep
        if input.vpot == 1
            A = VPstep(input.vtime, input.vamp, ii, input.tstep)
            VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
            VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
        elseif input.vpot == 2
            A = VPgaus(input.vtime, input.vamp, ii, input.tstep)
            VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
            VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
        elseif input.vpot == 3
            A = VPlaser(input.vtime, input.vamp, input.vfreq, ii, input.tstep)
            VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
            VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
        elseif input.vpot == 4
            A = VPconst(input.vtime, input.vamp, ii, input.tstep)
            VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
            VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
        end
    elseif input.vpot == 1
        A = input.vamp
        VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
        VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
    elseif input.vpot == 3
        A = VPlaser(input.vtime, input.vamp, input.vfreq, ii, input.tstep)
        VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
        VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
    elseif input.vpot == 4
        A = VPconst(input.vtime, input.vamp, ii, input.tstep)
        VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
        VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
    end


    if input.fxcalpha^2 + input.hartreeu^2 > 1e-9
        if xc_cache === nothing
            lrcxco1, dlrxco1 = buildxc(rion1, boxl, input.fxcalpha, input.fxcgamma, input.hartreeu, 2*input.nchain1)
            lrcxco2, dlrxco2 = buildxc(rion2, boxl, input.fxcalpha, input.fxcgamma, input.hartreeu, 2*input.nchain2)
        else
            lrcxco1, dlrxco1, lrcxco2, dlrxco2 = xc_cache[1], xc_cache[2], xc_cache[3], xc_cache[4]
        end
        Hxc(fock1i, rho1i, gspop1, 2*input.nchain1, lrcxco1)
        Hxc(fock2i, rho2i, gspop2, 2*input.nchain2, lrcxco2)
        if input.hubbardu^2 > 1e-9
            Honsite(fock1i, rho1i, gspop1, 2*input.nchain1, input.hubbardu)
            Honsite(fock2i, rho2i, gspop2, 2*input.nchain2, input.hubbardu)
        end
        if input.ehrenfest == 1
            velverlet(rion1, fion1, vion1, input, boxl, pottable, rho1i, dlrxc=dlrxco1, dint = dint1)
            velverlet(rion2, fion2, vion2, input, boxl, pottable, rho2i, dlrxc=dlrxco2, dint = dint2)
        end
    else
        if input.hubbardu^2 > 1e-9
            Honsite(fock1i, rho1i, gspop1, 2*input.nchain1, input.hubbardu)
            Honsite(fock2i, rho2i, gspop2, 2*input.nchain2, input.hubbardu)
        end
        if input.ehrenfest == 1
            velverlet(rion1, fion1, vion1, input, boxl, pottable, rho1i, dint = dint1)
            velverlet(rion2, fion2, vion2, input, boxl, pottable, rho2i, dint = dint2)
        end
    end

    if input.propflag == 0
        RungeCuta(rho1i,fock1i,input.tstep)
        RungeCuta(rho2i,fock2i,input.tstep)
    else
        densint1 = copy(rho1i)
        densint2 = copy(rho2i)
        commutator!(-1im*0.5*input.tstep, fock1i, rho1i, 1, densint1)
        commutator!(-1im*0.5*input.tstep, fock2i, rho2i, 1, densint2)
        construct_rdep_hamiltonian(fock1i, fock1aob, rion1, boxl, pottable, input, 2*input.nchain1)
        construct_rdep_hamiltonian(fock2i, fock2aob, rion2, boxl, pottable, input, 2*input.nchain2)
        Hint(fock1i, fock2i, densint1, densint2, gspop1, gspop2, 2*input.nchain1, 2*input.nchain2, intchm)

        if ii < input.vtime/input.tstep
            if input.vpot == 1
                A = VPstep(input.vtime, input.vamp, ii, input.tstep)
                VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
                VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
            elseif input.vpot == 2
                A = VPgaus(input.vtime, input.vamp, ii, input.tstep)
                VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
                VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
            elseif input.vpot == 3
                A = VPlaser(input.vtime, input.vamp, input.vfreq, ii, input.tstep)
                VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
                VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
            elseif input.vpot == 4
                A = VPconst(input.vtime, input.vamp, ii, input.tstep)
                VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
                VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
            end
        elseif input.vpot == 1
            A = input.vamp
            VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
            VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
        elseif input.vpot == 3
            A = VPlaser(input.vtime, input.vamp, input.vfreq, ii, input.tstep)
            VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
            VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
        elseif input.vpot == 4
            A = VPconst(input.vtime, input.vamp, ii, input.tstep)
            VectorPotMat(fock1i, fock1aob, A, 2*input.nchain1, rion1, boxl)
            VectorPotMat(fock2i, fock2aob, A, 2*input.nchain2, rion2, boxl)
        end

        if input.fxcalpha^2 + input.hartreeu^2 > 1e-9
            Hxc(fock1i, densint1, gspop1, 2*input.nchain1, lrcxco1)
            Hxc(fock2i, densint2, gspop2, 2*input.nchain2, lrcxco2)
        end
        if input.hubbardu^2 > 1e-9
            Honsite(fock1i, densint1, gspop1, 2*input.nchain1, input.hubbardu)
            Honsite(fock2i, densint2, gspop2, 2*input.nchain2, input.hubbardu)
        end

        if input.propflag == 1
            commutator!(-1im*input.tstep, fock1i, densint1, 1, rho1i)
            commutator!(-1im*input.tstep, fock2i, densint2, 1, rho2i)
        elseif input.propflag == 2
            rho1i .= exp(-1im * collect(fock1i) * input.tstep) * rho1i * exp(1im * collect(fock1i) * input.tstep)
            rho2i .= exp(-1im * collect(fock2i) * input.tstep) * rho2i * exp(1im * collect(fock2i) * input.tstep)
        end
    end
end

function construct_rion(nchain::Int64, req1::Float64, req2::Float64)
    rvect = zeros(Float64, 2*nchain)
    for ii = 1:nchain
        rvect[Int((ii-1)*2+1)] = (ii-1)*(req1+req2)
        rvect[Int(2*ii)] = (ii-1)*(req1+req2) + req1
    end
    return rvect
end

function construct_potential(input::Input)
    pottable = zeros(Float64, 10000, 5) # r, hop, rep, d/dr hop, d/dr rep1
    for jj = 1:10000
        r1              = (0.5 + 1.5 * jj / 10000) * input.req
        pottable[jj,1]  = r1
        pottable[jj,2]  = input.hop + input.hopslope * (r1 - input.req)
        pottable[jj,3]  = -1 * input.hop * input.pref * (input.req / r1)^input.p
        pottable[jj,4]  = input.hopslope
        pottable[jj,5]  = input.hop * input.pref * input.p * input.req^input.p / r1^(input.p+1)
    end
    return pottable
end

# Energy and dimerizing force of a closed nchain-cell ring with intra-cell bond r1 and
# inter-cell bond r2, evaluated from the continuous hop(r)/rep(r) laws (not the tabulated
# pottable) since the optimizer needs to resolve the small r1/r2 force imbalance near the
# minimum, well below pottable's grid resolution. Returns the *per-cell* (intensive) force
# conjugate to r1, i.e. -d(E/nchain)/dr1 at fixed r1+r2 (= -dE/dr1 once divided by nchain) --
# this is what must vanish at the equilibrium dimerization, independent of chain length.
function dimer_energy_force(nchain::Int64, r1::Float64, r2::Float64, input::Input)
    nsize = 2*nchain
    hop1 = input.hop + input.hopslope * (r1 - input.req)
    hop2 = input.hop + input.hopslope * (r2 - input.req)
    fockaob = construct_aob_hamiltonian(nchain, hop1, hop2)
    levels, solv = eigen(collect(fockaob))
    isolv = inv(solv)
    gap = abs(levels[Int(nsize/2)+1] - levels[Int(nsize/2)])
    rhogs = construct_gs_dens(solv, isolv, gap < 1e-8 ? 0.0 : 1.0)

    rep1 = -1 * input.hop * input.pref * (input.req / r1)^input.p
    rep2 = -1 * input.hop * input.pref * (input.req / r2)^input.p
    energy = real(2*sum(levels[1:Int(nsize/2)])) + nchain*(rep1 + rep2)

    rhoint1 = 2 * real(rhogs[1,2]) # bond order of an r1 (intra-cell) bond
    rhoint2 = 2 * real(rhogs[2,3]) # bond order of an r2 (inter-cell) bond
    drep1 = input.hop * input.pref * input.p * input.req^input.p / r1^(input.p+1)
    drep2 = input.hop * input.pref * input.p * input.req^input.p / r2^(input.p+1)
    dEdr1_percell = (input.hopslope*rhoint1 + drep1) - (input.hopslope*rhoint2 + drep2)

    return energy, -dEdr1_percell
end

# Steepest descent (with backtracking line search) on the dimerization coordinate r1
# (r2 = latcell - r1 always, so the total cell length is preserved exactly). A perfectly
# uniform starting guess (r1 == r2) sits at an unstable, symmetry-protected force=0 point
# for any Peierls-active chain, so it is nudged off-center before the first step.
function minimize_dimer(nchain::Int64, r1_0::Float64, r2_0::Float64, latcell::Float64, input::Input)
    r1 = r1_0
    r2 = r2_0
    if abs(r1 - r2) < 1e-6*latcell
        r1 += 1e-4*latcell
        r2 -= 1e-4*latcell
    end

    energy, force = dimer_energy_force(nchain, r1, r2, input)
    step = input.mingeomstep
    converged = false
    iters = 0

    for it = 1:input.mingeomiter
        iters = it
        if abs(force) < input.mingeomtol
            converged = true
            break
        end

        trial = step
        r1n = r1
        r2n = r2
        energyn = energy
        while true
            r1n = r1 + trial*force
            r2n = latcell - r1n
            if r1n <= 0.1*input.req || r2n <= 0.1*input.req
                energyn = energy + 1.0 # invalid geometry: reject and shrink
            else
                energyn, _ = dimer_energy_force(nchain, r1n, r2n, input)
            end
            if energyn <= energy || trial < 1e-12
                break
            end
            trial /= 2
        end

        r1, r2, energy = r1n, r2n, energyn
        step = trial*1.5
        _, force = dimer_energy_force(nchain, r1, r2, input)
    end

    return r1, r2, converged, iters, force
end

# Diagonalize a (Hermitian-valued) Fock matrix and refill the lowest half of the bands
# (closed shell, half filling), same fill rule as construct_gs_dens/dimer_energy_force:
# the two states straddling the Fermi level get occupation 1 each instead of 2/0 if they
# are degenerate (gap < 1e-8), e.g. an undimerized or not-yet-symmetry-broken trial.
function density_from_fock(fock::AbstractMatrix{ComplexF64})
    nsize = size(fock, 1)
    levels, solv = eigen(collect(fock))
    isolv = inv(solv)
    gap = abs(levels[Int(nsize/2)+1] - levels[Int(nsize/2)])
    rho = construct_gs_dens(solv, isolv, gap < 1e-8 ? 0.0 : 1.0)
    return rho, levels
end

# Self-consistent mean-field ground state for a single chain under its own Hxc kernel.
# The bare/decoupled chain's ground state (construct_gs_dens on fockaob alone) is only
# guaranteed to be a stationary point of the *interacting* Fock operator when the
# interaction is weak; this iterates the interacting problem to convergence instead of
# assuming that. A small random on-site potential is added to the Fock matrix's diagonal
# for the first `scfnoiseiter` iterations to break the exact symmetry that would otherwise
# pin the iteration exactly at an unstable saddle (same role as minimize_dimer's
# off-center nudge) -- removed afterwards so it doesn't bias the converged answer.
function scf_ground_state(fockaob::SparseMatrixCSC{Complex{Float64}}, rho_0::Array{Complex{Float64},2},
    gspop::Array{Complex{Float64},1}, lrcxco::Array{Complex{Float64},2}, input::Input)

    nsize = size(fockaob, 1)
    rho = copy(rho_0)
    converged = false
    resid = Inf
    it = 0

    for iter = 1:input.scfiter
        it = iter
        fockt = copy(fockaob)
        if iter <= input.scfnoiseiter
            for jj = 1:nsize
                fockt[jj,jj] += input.scfnoise * (rand() - 0.5)
            end
        end

        Hxc(fockt, rho, gspop, nsize, lrcxco)
        if input.hubbardu^2 > 1e-9
            Honsite(fockt, rho, gspop, nsize, input.hubbardu)
        end
        rhon, _ = density_from_fock(fockt)
        resid = maximum(abs.(diag(rhon) .- diag(rho)))
        rho .= (1 - input.scfmix) .* rho .+ input.scfmix .* rhon

        if iter > input.scfnoiseiter && resid < input.scftol
            converged = true
            break
        end
    end

    return rho, converged, it, resid
end

# Self-consistent mean-field ground state for two chains coupled by the interchain
# Coulomb term (Hint), in addition to each chain's own Hxc kernel. Hint is always active
# in two-chain mode regardless of fxcalpha/hartreeu, so this matters even with the
# Hartree/fxc kernels switched off. Same symmetry-breaking noise/mixing strategy as the
# single-chain overload above, applied jointly to both chains each iteration since they
# are coupled (not solved independently).
function scf_ground_state(fockaob1::SparseMatrixCSC{Complex{Float64}}, fockaob2::SparseMatrixCSC{Complex{Float64}},
    rho1_0::Array{Complex{Float64},2}, rho2_0::Array{Complex{Float64},2},
    gspop1::Array{Complex{Float64},1}, gspop2::Array{Complex{Float64},1},
    lrcxco1::Array{Complex{Float64},2}, lrcxco2::Array{Complex{Float64},2}, intchm::Array{Float64,2},
    input::Input)

    size1 = size(fockaob1, 1)
    size2 = size(fockaob2, 1)
    rho1 = copy(rho1_0)
    rho2 = copy(rho2_0)
    converged = false
    resid = Inf
    it = 0

    for iter = 1:input.scfiter
        it = iter
        fock1t = copy(fockaob1)
        fock2t = copy(fockaob2)
        if iter <= input.scfnoiseiter
            for jj = 1:size1
                fock1t[jj,jj] += input.scfnoise * (rand() - 0.5)
            end
            for jj = 1:size2
                fock2t[jj,jj] += input.scfnoise * (rand() - 0.5)
            end
        end

        Hxc(fock1t, rho1, gspop1, size1, lrcxco1)
        Hxc(fock2t, rho2, gspop2, size2, lrcxco2)
        Hint(fock1t, fock2t, rho1, rho2, gspop1, gspop2, size1, size2, intchm)
        if input.hubbardu^2 > 1e-9
            Honsite(fock1t, rho1, gspop1, size1, input.hubbardu)
            Honsite(fock2t, rho2, gspop2, size2, input.hubbardu)
        end

        rho1n, _ = density_from_fock(fock1t)
        rho2n, _ = density_from_fock(fock2t)
        resid = max(maximum(abs.(diag(rho1n) .- diag(rho1))), maximum(abs.(diag(rho2n) .- diag(rho2))))

        rho1 .= (1 - input.scfmix) .* rho1 .+ input.scfmix .* rho1n
        rho2 .= (1 - input.scfmix) .* rho2 .+ input.scfmix .* rho2n

        if iter > input.scfnoiseiter && resid < input.scftol
            converged = true
            break
        end
    end

    return rho1, rho2, converged, it, resid
end

