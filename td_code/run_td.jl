using LinearAlgebra
using SparseArrays

include("TLS_module.jl")
include("units.jl")
include("IOmodule.jl")

###################
# READ INPUT FILE #
###################

input = read_input_file("inp.in")
write_header(input)

######################
# SYSTEM CONSRUCTION #
######################

if input.chains == 1 # Single chain system definition
  pottable          = construct_potential(input)
  r1init            = input.lattice * (1 - input.dimer1) / 2
  r2init            = input.lattice * (1 + input.dimer1) / 2

  if input.mingeom == 1 && input.restart == 0 # relax the dimerization to its minimum-energy value
    r1init, r2init, mgconv, mgiters, mgforce = minimize_dimer(input.nchain1, r1init, r2init, input.lattice, input)
    input.dimer1 = (r2init - r1init) / input.lattice
    println("Geometry optimization: ", mgconv ? "converged" : "did NOT converge", " after $mgiters iterations (dimer1 = $(input.dimer1))")
    write_mingeom("chain 1", mgconv, mgiters, input.dimer1, mgforce)
  end

  hi1               = pottable[Int64(round((r1init/input.req - 0.5)*10000/1.5)), 2]
  hi2               = pottable[Int64(round((r2init/input.req - 0.5)*10000/1.5)), 2]
  fockaob           = construct_aob_hamiltonian(input.nchain1, hi1, hi2)
  levels, solv      = eigen(collect(fockaob))
  isolv             = inv(solv)
  rhogs             = construct_gs_dens(solv, isolv, input.dimer1)
  gspop             = diag(rhogs)
  if input.restart == 1
    rhogs, rion, vion = read_restart("restart.dat", 2*input.nchain1)
  else
    rion              = construct_rion(input.nchain1, r1init, r2init) #Due to different forces the equilibrium geometry in the ground state is r1 = 1.9579 AA r2 = 2.5453 AA
    vion              = zeros(Float64, 2*input.nchain1)
  end
  fion              = zeros(Float64, 2*input.nchain1)
  boxl              = input.nchain1 * input.lattice

  write_eigen(levels)

elseif  input.chains == 2 # Double chain system definition
  pottable          = construct_potential(input)
  r1init1            = input.lattice * (1 - input.dimer1) / 2
  r2init1            = input.lattice * (1 + input.dimer1) / 2
  latcell2           = input.lattice * input.nchain1 / input.nchain2
  r1init2            = latcell2 * (1 - input.dimer2) / 2
  r2init2            = latcell2 * (1 + input.dimer2) / 2

  if input.mingeom == 1 && input.restart == 0 # relax each chain's dimerization to its minimum-energy value
    r1init1, r2init1, mgconv1, mgiters1, mgforce1 = minimize_dimer(input.nchain1, r1init1, r2init1, input.lattice, input)
    input.dimer1 = (r2init1 - r1init1) / input.lattice
    println("Geometry optimization (chain 1): ", mgconv1 ? "converged" : "did NOT converge", " after $mgiters1 iterations (dimer1 = $(input.dimer1))")
    write_mingeom("chain 1", mgconv1, mgiters1, input.dimer1, mgforce1)

    r1init2, r2init2, mgconv2, mgiters2, mgforce2 = minimize_dimer(input.nchain2, r1init2, r2init2, latcell2, input)
    input.dimer2 = (r2init2 - r1init2) / latcell2
    println("Geometry optimization (chain 2): ", mgconv2 ? "converged" : "did NOT converge", " after $mgiters2 iterations (dimer2 = $(input.dimer2))")
    write_mingeom("chain 2", mgconv2, mgiters2, input.dimer2, mgforce2)
  end

  hi11               = pottable[Int64(round((r1init1/input.req - 0.5)*10000/1.5)), 2]
  hi21               = pottable[Int64(round((r2init1/input.req - 0.5)*10000/1.5)), 2]
  fockaob1          = construct_aob_hamiltonian(input.nchain1, hi11, hi21)
  levels1, solv1    = eigen(collect(fockaob1))
  isolv1            = inv(solv1)
  rho1gs            = construct_gs_dens(solv1, isolv1, input.dimer1)
  gspop1            = diag(rho1gs)


  write_eigen(levels1)

  hi12               = pottable[Int64(round((r1init2/input.req - 0.5)*10000/1.5)), 2]
  hi22               = pottable[Int64(round((r2init2/input.req - 0.5)*10000/1.5)), 2]
  fockaob2           = construct_aob_hamiltonian(input.nchain2, hi12, hi22)
  levels2, solv2     = eigen(collect(fockaob2))
  isolv2             = inv(solv2)
  rho2gs             = construct_gs_dens(solv2, isolv2, input.dimer2)
  gspop2             = diag(rho2gs)
  
  write_eigen(levels2)

  if input.restart == 1
    rho1gs, rho2gs, rion1, rion2, vion1, vion2 = read_restart("restart.dat", 2*input.nchain1, 2*input.nchain2)
  else
    rion1              = construct_rion(input.nchain1, r1init1, r2init1) #Due to different forces the equilibrium geometry in the ground state is r1 = 1.9579 AA r2 = 2.5453 AA
    vion1              = zeros(Float64, 2*input.nchain1)
    rion2              = construct_rion(input.nchain2, r1init2, r2init2) #Due to different forces the equilibrium geometry in the ground state is r1 = 1.9579 AA r2 = 2.5453 AA
    vion2              = zeros(Float64, 2*input.nchain2)
  end
  fion1             = zeros(Float64, 2*input.nchain1)
  fion2             = zeros(Float64, 2*input.nchain2)
  boxl              = input.nchain1 * input.lattice

end


######################
# BUFFER DECLARATION #
######################

tbuffer = Float64[]
eionbuffer1 = Float64[]
eionbuffer2 = Float64[]
ebuffer1 = Float64[]
ebuffer2 = Float64[]
cbuffer1 = Float64[]
cbuffer2 = Float64[]

##################
# TD PORPAGATION #
##################

write_td_header(input)

if input.chains == 1 # Single chain system propagation
  focki     = copy(fockaob)
  rhoi      = copy(rhogs)
  # With ehrenfest=0 the ions never move, so the Hartree/fxc interaction matrix (buildxc) is the
  # same on every step; precompute it once instead of paying its O(nchain1^2) cost every step.
  xc_cache  = input.ehrenfest == 0 ? buildxc(rion, boxl, input.fxcalpha, input.fxcgamma, input.hartreeu, 2*input.nchain1) : nothing
  for ii = 1:input.steps
    propagate(rhoi, gspop, focki, fockaob, rion, vion, fion, boxl, pottable, input, ii; xc_cache=xc_cache) #time evolve rhoi and focki

    if (ii % input.savefreq) == 0 #save to buffer
      push!(tbuffer, ii*input.tstep)
      push!(ebuffer1, real(tr(focki*rhoi)-0.5*sum(diag(focki).*diag(rhoi.-0.5))))
      push!(eionbuffer1, ionenergy(rion, vion, input.atmass, boxl, pottable))
      push!(cbuffer1, 2 * imag(rhoi[1,2]*focki[2,1]))
      write_frame(rion, real.(diag(rhoi).-gspop).+2.0,ii, boxl)
    end
    if ii % (input.savefreq*input.bufflen) == 0 #write buffer to file
      write_buffer(tbuffer, ebuffer1, eionbuffer1, cbuffer1, input.bufflen)
      empty!(tbuffer)
      empty!(ebuffer1)
      empty!(eionbuffer1)
      empty!(cbuffer1)
      write_restart(rhoi, rion, vion)
    end
  end
  if !isempty(tbuffer) #write buffer leftover to file
    write_buffer(tbuffer, ebuffer1, eionbuffer1, cbuffer1, length(tbuffer))
    write_restart(rhoi, rion, vion)
  end
end

if input.chains == 2 # Double chain system propagation
  fock1i    = copy(fockaob1)
  rho1i     = copy(rho1gs)
  fock2i    = copy(fockaob2)
  rho2i     = copy(rho2gs)
  # With ehrenfest=0 neither chain's ions move, so buildxc/buildintchain are the same on every
  # step; precompute them once instead of paying their O(size^2) cost every step.
  if input.ehrenfest == 0
    lrcxco1, dlrxco1 = buildxc(rion1, boxl, input.fxcalpha, input.fxcgamma, input.hartreeu, 2*input.nchain1)
    lrcxco2, dlrxco2 = buildxc(rion2, boxl, input.fxcalpha, input.fxcgamma, input.hartreeu, 2*input.nchain2)
    intchm0, dintchm0 = buildintchain(rion1, rion2, boxl, input)
    xc_cache = (lrcxco1, dlrxco1, lrcxco2, dlrxco2, intchm0, dintchm0)
  else
    xc_cache = nothing
  end
  for ii = 1:input.steps
    propagate(rho1i, rho2i,
    gspop1, gspop2,
    fock1i, fock2i,
    fockaob1, fockaob2,
    rion1, rion2,
    vion1, vion2,
    fion1, fion2,
    boxl, pottable,
    input, ii; xc_cache=xc_cache) #time evolve rhoi and focki

    if (ii % input.savefreq) == 0 #save to buffer
      push!(tbuffer, ii*input.tstep)
      push!(ebuffer1, real(tr(fock1i*rho1i)-0.5*sum(diag(fock1i).*(diag(rho1i).-gspop1))))
      push!(ebuffer2, real(tr(fock2i*rho2i)-0.5*sum(diag(fock2i).*(diag(rho2i).-gspop2))))
      push!(eionbuffer1, ionenergy(rion1, vion1, input.atmass, boxl, pottable))
      push!(eionbuffer2, ionenergy(rion2, vion2, input.atmass, boxl, pottable))
      push!(cbuffer1, 2 * imag(rho1i[1,2]*fock1i[2,1]))
      push!(cbuffer2, 2 * imag(rho2i[1,2]*fock2i[2,1]))
      write_frame(rion1, rion2, real.(diag(rho1i).-gspop1), real.(diag(rho2i).-gspop2), ii, boxl, input.rchain)
    end
    if ii % (input.savefreq*input.bufflen) == 0 #write buffer to file
      write_buffer(tbuffer, ebuffer1, ebuffer2, eionbuffer1, eionbuffer2, cbuffer1, cbuffer2, input.bufflen)
      empty!(tbuffer)
      empty!(ebuffer1)
      empty!(ebuffer2)
      empty!(eionbuffer1)
      empty!(eionbuffer2)
      empty!(cbuffer1)
      empty!(cbuffer2)
      write_restart(rho1i, rho2i, rion1, rion2, vion1, vion2)
    end
  end
  if !isempty(tbuffer) #write buffer leftover to file
    write_buffer(tbuffer, ebuffer1, ebuffer2, eionbuffer1, eionbuffer2, cbuffer1, cbuffer2, length(tbuffer))
    write_restart(rho1i, rho2i, rion1, rion2, vion1, vion2)
  end
end
