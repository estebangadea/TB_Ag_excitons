using Printf
using Serialization

include("units.jl")

Base.@kwdef mutable struct Input
  # ELECTRONIC SYSTEM 
  chains::Int64       = -1 #number of chains: 1 or 2
  nchain1::Int64      = -1 #number of pairs in chain 1
  nchain2::Int64      = -1 #number of pairs in chain 2
  hop::Float64        = -0.0245725447 #hopping integral 1
  hopslope::Float64   = 0.007215487659 #slope d(hop)/dr of the hopping-vs-distance relation
  req::Float64        = 4.922388 #equilibrium distance between closest atoms (hopping integral 1)
  lattice::Float64    = 11.3383567477 #lattice distance
  dimer1::Float64     = 0.0 #dimerization of chain1
  dimer2::Float64     = 0.0 #dimerization of chain2
  rchain::Float64     = 10 #intechain distance
  hartreeu::Float64   = 0.1152918 #strength of the Hartree term (U); on-site value and 1/U decay-softening radius, K=1 implicit
  fxcalpha::Float64   = -0.0073498645 #strenght of the fxc interaction
  fxcgamma::Float64   = 0.4922388 #radius for long-range correction
  p::Int64            =  15     #exponent of repulsion potential
  pref::Float64       =  0.231122  #Prefactor of repulsion potential

  # NUCLEAR SYSTEM
  atmass::Float64     = 198046 #mass of the atoms
  damp::Int64         = 0 #0-no damp, 1-velocity damp
  dampforce::Float64  = 0.1 #dump force constant
  ehrenfest::Int64    = 0 #0-frozen ions (electronic dynamics only), 1-Ehrenfest dynamics (ions move)

  # GEOMETRY OPTIMIZATION
  mingeom::Int64       = 0 #0-skip, 1-steepest-descent relax dimer1/dimer2 to the minimum-energy value before propagation (ignored if restart=1)
  mingeomtol::Float64  = 1e-5*ELECTRONVOLT/ANGSTROM #convergence threshold on the residual dimerizing force (per cell)
  mingeomstep::Float64 = 50.0 #initial steepest-descent step size (a.u.), refined every iteration by backtracking line search
  mingeomiter::Int64   = 2000 #maximum number of steepest-descent iterations

  # TIME PROPAGATION
  time::Float64       = 0.005 #total time in atomic units
  tstep::Float64      = 0.005 #timestep in atomic units
  steps::Int64        = 1 #number of steps
  savefreq::Int64     = 100 #frequency of data save
  bufflen::Int64      = 1000 #size of the data buffer
  restart::Int64      = 0 #0-from scratch, 1-read restart.dat
  propflag::Int64     = 0 #0-RC, 1-RCsc, 2-UnitProp 

  # VECTOR POTENTIAL
  vpot::Int64         = 0 #vector potential type: 0-None 1-step 2-pulse 3-laser pulse 4-ramp (smooth turn-on to constant field)
  vamp::Float64       = 0.0 #amplitud of the vector potential
  vtime::Float64      = 0.005 #time of the aplication
  vfreq::Float64      = 1.0 #frequency of the laser (only if vpot = 3)

end

function read_input_file(filename)
  try
      # Initialize an empty Config struct
      input = Input()

      # Read the file and split it into lines
      lines = readlines(filename)

      # Process each line to extract key-value pairs
      for line in lines
          parts = split(line, '=')
          if length(parts) > 1
            key = strip(parts[1])
            value = strip(split(parts[2])[1])

            # Update the config struct based on the key
            if key == "chains"
                input.chains = parse(Int64, value)
            elseif key == "nchain1"
                input.nchain1 = parse(Int64, value)
            elseif key == "nchain2"
                input.nchain2 = parse(Int64, value)
            elseif key == "hop"
                input.hop = parse(Float64, value) * ELECTRONVOLT
            elseif key == "hopslope"
                input.hopslope = parse(Float64, value) * ELECTRONVOLT / ANGSTROM
            elseif key == "req"
                input.req = parse(Float64, value) * ANGSTROM
            elseif key == "lattice"
              input.lattice = parse(Float64, value) * ANGSTROM
            elseif key == "dimer1"
              input.dimer1 = parse(Float64, value)
            elseif key == "dimer2"
              input.dimer2 = parse(Float64, value)
            elseif key == "rchain"
                input.rchain = parse(Float64, value) * ANGSTROM
            elseif key == "hartreeu"
                input.hartreeu = parse(Float64, value) * ELECTRONVOLT
            elseif key == "fxcalpha"
                input.fxcalpha = parse(Float64, value) * ELECTRONVOLT
            elseif key == "fxcgamma"
                input.fxcgamma = parse(Float64, value) * ANGSTROM
            elseif key == "p"
                input.p = parse(Int64, value)
            elseif key == "pref"
                input.pref = parse(Float64, value)
            elseif key == "atmass"
                input.atmass = parse(Float64, value)
            elseif key == "damp"
                input.damp = parse(Int64, value)
              elseif key == "dampforce"
                input.dampforce = parse(Float64, value)
            elseif key == "ehrenfest"
                input.ehrenfest = parse(Int64, value)
            elseif key == "mingeom"
                input.mingeom = parse(Int64, value)
            elseif key == "mingeomtol"
                input.mingeomtol = parse(Float64, value) * ELECTRONVOLT / ANGSTROM
            elseif key == "mingeomstep"
                input.mingeomstep = parse(Float64, value)
            elseif key == "mingeomiter"
                input.mingeomiter = parse(Int64, value)
            elseif key == "time"
                input.time = parse(Float64, value) * FEMTOSECOND
            elseif key == "tstep"
                input.tstep = parse(Float64, value) * FEMTOSECOND
                input.steps = Int64(floor((input.time / input.tstep)))
            elseif key == "savefreq"
                input.savefreq = parse(Int64, value)
            elseif key == "bufflen"
                input.bufflen = parse(Int64, value)
            elseif key == "restart"
                input.restart = parse(Int64, value)
            elseif key == "propflag"
                input.propflag = parse(Int64, value)
            elseif key == "vpot"
                input.vpot = parse(Int64, value)
            elseif key == "vamp"
                input.vamp = parse(Float64, value) * ELECTRONVOLT / ANGSTROM
            elseif key == "vtime"
                input.vtime = parse(Float64, value) * FEMTOSECOND
            elseif key == "vfreq"
                input.vfreq = parse(Float64, value) * ELECTRONVOLT
            end
          end
      end

      return input

  catch e
      println("An error occurred: $e")
      return nothing
  end
end

function write_header(input::Input)
  open("output.out", "w") do io
    write(io, "Tigh binding simulation of closed 1D chains\n\n")
    write(io, "number of chains     = $(input.chains)\n")
    if input.chains == 1
      write(io, "number of pairs      = $(input.nchain1)\n")
    elseif input.chains == 2
      write(io, "pairs in chain 1     = $(input.nchain1)\n")
      write(io, "pairs in chain 1     = $(input.nchain2)\n")
      write(io, @sprintf("interchain distance  = %.6f AA\n", input.rchain/ANGSTROM))
    end
    write(io, @sprintf("hopping term       = %.6f eV\n", input.hop/ELECTRONVOLT))
    write(io, @sprintf("hopping slope      = %.6f eV/AA\n", input.hopslope/ELECTRONVOLT*ANGSTROM))
    write(io, @sprintf("eq distance        = %.6f AA\n", input.req/ANGSTROM))

    write(io, @sprintf("hartree U            = %.6f eV\n", input.hartreeu/ELECTRONVOLT))
    write(io, @sprintf("fxc alpha            = %.6f eV\n", input.fxcalpha/ELECTRONVOLT))
    write(io, @sprintf("fxc gamma            = %.6f AA\n\n", input.fxcgamma/ANGSTROM))

    write(io, "Time propagation\n\n")
    write(io, @sprintf("total time           = %.6f fs\n", input.time/FEMTOSECOND))
    write(io, @sprintf("time step            = %.6f fs\n\n", input.tstep/FEMTOSECOND))
    if input.vpot == 0
      write(io, "No vector potential\n")
    elseif input.vpot == 1
      write(io, "Vector potential step\n")
      write(io, @sprintf("amplitud           = %.6f eV/AA\n", input.vamp/ELECTRONVOLT * ANGSTROM))
      write(io, @sprintf("duration           = %.6f fs\n\n", input.vtime/FEMTOSECOND))
    elseif input.vpot == 2
      write(io, "Vector potential pulse\n")
      write(io, @sprintf("amplitud           = %.6f eV/AA\n", input.vamp/ELECTRONVOLT * ANGSTROM))
      write(io, @sprintf("duration           = %.6f fs\n\n", input.vtime/FEMTOSECOND))
    elseif input.vpot == 3
      write(io, "Vector potential laser pulse\n")
      write(io, @sprintf("amplitud           = %.6f eV/AA\n", input.vamp/ELECTRONVOLT * ANGSTROM))
      write(io, @sprintf("duration           = %.6f fs\n", input.vtime/FEMTOSECOND))
      write(io, @sprintf("frequency          = %.6f eV\n\n", input.vfreq/ELECTRONVOLT))
    elseif input.vpot == 4
      write(io, "Vector potential ramp (smooth turn-on to constant field)\n")
      write(io, @sprintf("amplitud           = %.6f eV/AA\n", input.vamp/ELECTRONVOLT * ANGSTROM))
      write(io, @sprintf("duration           = %.6f fs\n\n", input.vtime/FEMTOSECOND))
    end
  end
  open("traj.lammpstrj", "w") do io
  end
end

function write_mingeom(label::String, converged::Bool, iters::Int64, dimer::Float64, force::Float64)
  open("output.out", "a") do io
    write(io, @sprintf("Geometry optimization (%s): %s after %d iterations\n",
      label, converged ? "converged" : "did NOT converge", iters))
    write(io, @sprintf("  optimized dimerization = %.6f, residual force = %.6e eV/AA\n\n",
      dimer, force/ELECTRONVOLT*ANGSTROM))
  end
end

function write_eigen(values::Vector{Float64})
  open("output.out", "a") do io
    write(io, "Eigenvalues:\n")
    for val in values
      write(io, @sprintf("%14.8f\n", val/ELECTRONVOLT))
    end
    write(io, "\n")
  end
end

function write_buffer(tbuff::Vector{Float64}, ebuff::Vector{Float64}, iebuff::Vector{Float64}, cbuff::Vector{Float64}, len::Int64)
  open("output.out", "a") do io
      for ii = 1:len
        write(io, @sprintf("%14.6f  %22.8f  %22.8f  %18.8e\n",
          tbuff[ii]/FEMTOSECOND, ebuff[ii]/ELECTRONVOLT, iebuff[ii]/ELECTRONVOLT, cbuff[ii]*FEMTOSECOND))
      end
  end
end

function write_buffer(tbuff::Vector{Float64}, ebuff1::Vector{Float64}, ebuff2::Vector{Float64}, iebuff1::Vector{Float64}, iebuff2::Vector{Float64}, cbuff1::Vector{Float64}, cbuff2::Vector{Float64}, len::Int64)
  open("output.out", "a") do io
      for ii = 1:len
        write(io, @sprintf("%14.6f  %23.8f  %23.8f  %20.8f  %20.8f  %18.8e  %18.8e\n",
          tbuff[ii]/FEMTOSECOND, ebuff1[ii]/ELECTRONVOLT, ebuff2[ii]/ELECTRONVOLT,
          iebuff1[ii]/ELECTRONVOLT, iebuff2[ii]/ELECTRONVOLT, cbuff1[ii]*FEMTOSECOND, cbuff2[ii]*FEMTOSECOND))
      end
  end
end

function write_frame(rion::Array{Float64,1}, pops::Array{Float64,1}, ii::Int64, boxl::Float64)
  size = Int64(length(rion))
  open("traj.lammpstrj", "a") do io
    write(io, "ITEM: TIMESTEP\n$(ii)\n")
    write(io, "ITEM: NUMBER OF ATOMS\n$(size)\n")
    write(io, @sprintf("ITEM: BOX BOUNDS pp pp pp\n%.6f %.6f\n%.6f %.6f\n%.6f %.6f\n",
      0.0, boxl/ANGSTROM, -100.0, 100.0, -100.0, 100.0))
    write(io, "ITEM: ATOMS id type x y z vx vy vz\n")
      for ii = 1:size
        write(io, @sprintf("%6d %3d %14.6f %14.6f %14.6f %12.6f %12.6f %12.6f\n",
          ii, 1, rion[ii]/ANGSTROM, 0.0, 0.0, pops[ii], 0.0, 0.0))
      end
  end
end

function write_frame(rion1::Array{Float64,1}, rion2::Array{Float64,1}, pops1::Array{Float64,1}, pops2::Array{Float64,1}, ii::Int64, boxl::Float64, distc::Float64)
  size1 = Int64(length(rion1))
  size2 = Int64(length(rion2))
  open("traj.lammpstrj", "a") do io
    write(io, "ITEM: TIMESTEP\n$(ii)\n")
    write(io, "ITEM: NUMBER OF ATOMS\n$(size1+size2)\n")
    write(io, @sprintf("ITEM: BOX BOUNDS pp pp pp\n%.6f %.6f\n%.6f %.6f\n%.6f %.6f\n",
      0.0, boxl/ANGSTROM, -100.0, 100.0, -100.0, 100.0))
    write(io, "ITEM: ATOMS id type x y z vx vy vz\n")
      for ii = 1:size1
        write(io, @sprintf("%6d %3d %14.6f %14.6f %14.6f %12.6f %12.6f %12.6f\n",
          ii, 1, rion1[ii]/ANGSTROM, 0.0, 0.0, pops1[ii], 0.0, 0.0))
      end
      for ii = 1:size2
        write(io, @sprintf("%6d %3d %14.6f %14.6f %14.6f %12.6f %12.6f %12.6f\n",
          size1+ii, 1, rion2[ii]/ANGSTROM, distc/ANGSTROM, 0.0, pops2[ii], 0.0, 0.0))
      end
  end
end

function write_td_header(input::Input)
  if input.chains == 1
    open("output.out", "a") do io
        write(io, @sprintf("%14s  %22s  %22s  %18s\n",
          "time(fs)", "electronic_energy(eV)", "nuclear_energy(eV)", "current(e/fs)"))
    end
  elseif input.chains ==2
    open("output.out", "a") do io
      write(io, @sprintf("%14s  %23s  %23s  %20s  %20s  %18s  %18s\n",
        "time(fs)", "electronic_energy1(eV)", "electronic_energy2(eV)",
        "nuclear_energy1(eV)", "nuclear_energy2(eV)", "current1(e/fs)", "current2(e/fs)"))
    end
  end
end

function write_restart(matrix::Array{ComplexF64,2}, array1::Array{Float64,1}, array2::Array{Float64,1})
  try
      open("restart.dat", "w") do file
          serialize(file, matrix)
          serialize(file, array1)
          serialize(file, array2)
      end

      println("Restart successfully written to restart.dat")

  catch e
      println("An error occurred: $e")
  end
end

function write_restart(matrix1::Array{ComplexF64,2}, matrix2::Array{ComplexF64,2}, rion1::Array{Float64,1}, rion2::Array{Float64,1}, vion1::Array{Float64,1}, vion2::Array{Float64,1})
  try
      open("restart.dat", "w") do file
          serialize(file, matrix1)
          serialize(file, rion1)
          serialize(file, vion1)
          serialize(file, matrix2)
          serialize(file, rion2)
          serialize(file, vion2)
      end

      println("Restart successfully written to restart.dat")

  catch e
      println("An error occurred: $e")
  end
end

function read_restart(filename, sizei)
  try
      matrix, array1, array2 = open(filename, "r") do file
          (deserialize(file), deserialize(file), deserialize(file))
      end

      return matrix, array1, array2

  catch e
      println("An error occurred: $e")
      return nothing, nothing, nothing
  end
end

function read_restart(filename, size1, size2)
  try
      matrix1, array1, array2, matrix2, array3, array4 = open(filename, "r") do file
          (deserialize(file), deserialize(file), deserialize(file),
           deserialize(file), deserialize(file), deserialize(file))
      end

      return matrix1, matrix2, array1, array3, array2, array4

  catch e
      println("An error occurred: $e")
      return nothing, nothing, nothing, nothing, nothing, nothing
  end
end