
using KomaMRICore, KomaMRIPlots, CUDA, PlotlyJS # Essentials

using Suppressor, ProgressLogging # Extras


sys = Scanner()
sys.B0 = 0.55
sys.Gmax = 40.0e-3
sys.Smax = 25.0e-3




# General sequence parameters
Trf = 500e-6  			# 500 [ms]
B1 = 1 / (360*γ*Trf)    # B1 amplitude [uT]
Tadc = 1e-6 			# 1us

# Prepulses
Tfatsat = 26.624e-3 # 26.6 [ms] FatSat duration
T2prep_duration = 50e-3 # 50 [ms]

# Acquisition
RR = 1.3 				# 1 [s]
dummy_heart_beats = 3 	# Steady-state
TR = 7e-3             # 5.3 [ms] RF Low SAR
#TE = TR / 2 			# bSSFP condition
TE = 3.51e-3 			# 
iNAV_lines = 2          # FatSat-Acq delay: iNAV_lines * TR
#iNAV_flip_angle = 3.2   # 3.2 [deg]
iNAV_flip_angle = 0.0
im_segments = 30        # Acquisitino window: im_segments * TR

# To be optimized
im_flip_angle = [110, 80] # 80 [deg]
FatSat_flip_angle = 180   # 180 [deg]
IR_inversion_time = 90e-3 # 90 [ms] 

seq_params = (;
    dummy_heart_beats,
    iNAV_lines,
    im_segments,
    iNAV_flip_angle,
    im_flip_angle,
    T2prep_duration,
    IR_inversion_time,
    RR,
    FatSat_flip_angle
)

seq_params

fat_ppm = -3.4e-6 			# -3.4ppm fat-water frequency shift
Niso = 200        			# 200 isochromats in spoiler direction
Δx_voxel = 1.0e-3 			# 1.0 [mm]
fat_freq = γ*sys.B0*fat_ppm # -80 [Hz]
dx = Array(range(-Δx_voxel/2, Δx_voxel/2, Niso))

function FatSat(α, Δf; sample=false)
    # FatSat design
    # cutoff_freq = sqrt(log(2) / 2) / a where B1(t) = exp(-(π t / a)^2)
    cutoff = fat_freq / π 			      # cutoff [Hz] => ≈1/10 RF power to water
    a = sqrt(log(2) / 2) / cutoff         # a [s]
    τ = range(-Tfatsat/2, Tfatsat/2, 64)  # time [s]
    gauss_pulse = exp.(-(π * τ / a) .^ 2) # B1(t) [T]
    # FatSat prepulse
    seq = Sequence()
    seq += Grad(-8e-3, 3000e-6, 500e-6) #Spoiler1
    seq += RF(gauss_pulse, Tfatsat, Δf)
    α_ref = get_flip_angles(seq)[2]
    seq *= (α/α_ref+0im)
    if sample
        seq += ADC(1, 1e-6)
    end
    seq += Grad(8e-3, 3000e-6, 500e-6) #Spoiler2
    if sample
        seq += ADC(1, 1e-6)
    end
    return seq
end

function T2prep(TE; sample=false)
    seq = Sequence()
    seq += RF(90 * B1, Trf)
    seq += sample ? ADC(20, TE/2 - 1.5Trf) : Delay(TE/2 - 1.5Trf)
    seq += RF(180im * B1 / 2, Trf*2)
    seq += sample ? ADC(20, TE/2 - 1.5Trf) : Delay(TE/2 - 1.5Trf)
    seq += RF(-90 * B1, Trf)
    seq += Grad(8e-3, 6000e-6, 600e-6) #Spoiler3
    if sample
        seq += ADC(1, 1e-6)
    end
    return seq
end

function IR(IR_delay; sample=false)
    # Generating HS pulse
    # Based on: https://onlinelibrary.wiley.com/doi/epdf/10.1002/jmri.26021
    # Params
    flip_angle = 900;    # Peak amplitude (deg)
    Trf = 10240e-6;      # Pulse duration (ms)
    β = 6.7e2;           # frequency modulation param (rad/s)
    μ = 5;               # phase modulation parameter (dimensionless)
    fmax = μ * β / (2π); # 2fmax = BW
    # RF pulse
    t = range(-Trf/2, Trf/2, 201);
    B1 = sech.(β .* t);
    Δf = fmax  .* tanh.(β .* t);
    # Spoiler length
    spoiler_time = 6000e-6
    spoiler_rise_fall = 600e-6
    # Prepulse
    seq = Sequence()
    seq += RF(B1, Trf, Δf) # FM modulated pulse
    seq = (flip_angle / get_flip_angles(seq)[1] + 0.0im) * seq # RF scaling
    seq += Grad(8e-3, spoiler_time, spoiler_rise_fall) #Spoiler3
    if sample
        seq += ADC(11, IR_delay - spoiler_time - 2spoiler_rise_fall)
    else
        seq += Delay(IR_delay - spoiler_time - 2spoiler_rise_fall)
    end
    return seq
end

function bSSFP(iNAV_lines, im_segments, iNAV_flip_angle, im_flip_angle; sample=false)
    k = 0
    seq = Sequence()
    for i = 0 : iNAV_lines + im_segments - 1
        if iNAV_lines != 0
            m = (im_flip_angle - iNAV_flip_angle) / iNAV_lines
            α = min( m * i + iNAV_flip_angle, im_flip_angle ) * (-1)^k
        else
            α = im_flip_angle * (-1)^k
        end
        seq += RF(α * B1, Trf)
        if i < iNAV_lines && !sample
            seq += Delay(TR - Trf)
        else
            seq += Delay(TE - Trf/2 - Tadc/2)
            seq += ADC(1, Tadc)
            seq += Delay(TR - TE - Tadc/2 - Trf/2)
        end
        k += 1
    end
    return seq
end


function BOOST(
            dummy_heart_beats,
            iNAV_lines,
            im_segments,
            iNAV_flip_angle,
            im_flip_angle,
            T2prep_duration,
            IR_inversion_time,
            
            RR,
            FatSat_flip_angle=180;
            sample_recovery=zeros(Bool, dummy_heart_beats+1)
            
            )
    # Seq init
    seq = Sequence()
    for hb = 1 : dummy_heart_beats + 1
        sample = sample_recovery[hb] # Sampling recovery curve for hb
        # Generating seq blocks
        t2p = T2prep(T2prep_duration; sample)
        ir = IR(IR_inversion_time - iNAV_lines * TR - Trf - TE; sample)
        fatsat = FatSat(FatSat_flip_angle, fat_freq; sample)
        # Magnetization preparations
        for contrast = 1:2
            preps = Sequence()
            if contrast == 1 # Bright-blood contrast
                preps += t2p
                preps += ir
            else # Reference contrast
                preps += fatsat
            end
            # Contrast dependant flip angle
            bssfp = bSSFP(iNAV_lines, im_segments, iNAV_flip_angle, im_flip_angle[contrast]; sample)
            # Concatenating seq blocks
            seq += preps
            seq += bssfp
            # RR interval consideration
            RRdelay = RR  - dur(bssfp) - dur(preps)
            seq += sample ? ADC(80, RRdelay) : Delay(RRdelay)
        end
    end
    return seq
end

function carotid_phantom(off; off_fat=fat_freq)
    carotid = Phantom{Float64}(x=dx, ρ=0.6*ones(Niso), T1=750e-3*ones(Niso),
                                T2=90e-3*ones(Niso),    Δw=2π*off*ones(Niso))
    blood =   Phantom{Float64}(x=dx, ρ=0.7*ones(Niso), T1=1122e-3*ones(Niso),
                                T2=263e-3*ones(Niso),   Δw=2π*off*ones(Niso))
    fat1 =    Phantom{Float64}(x=dx, ρ=1.0*ones(Niso), T1=183e-3*ones(Niso),
                                T2=93e-3*ones(Niso),    Δw=2π*(off_fat + off)*ones(Niso))
    obj = carotid + blood + fat1
    return obj
end
	

sim_params = Dict{String,Any}(
	"return_type"=>"mat",
	"sim_method"=>BlochDict(save_Mz=true),
	"Δt_rf"=>Trf,
	"gpu"=>false,
	"Nthreads"=>1
)

seq = BOOST(seq_params...; sample_recovery=ones(Bool, dummy_heart_beats+1))
obj = carotid_phantom(0)
magnetization = @suppress simulate(obj, seq, sys; sim_params)
nothing # hide output

plot_seq(seq; range=[5990, 6280], slider=true)

#plot_seq(seq; range=[6900, 7190], slider=true)

# Prep plots
labs = ["Carotid", "Blood", "Fat"]
cols = ["blue", "red", "green"]
spin_group = [(1:Niso)', (Niso+1:2Niso)', (2Niso+1:3Niso)']
t = KomaMRICore.get_adc_sampling_times(seq)
Mxy(i) = abs.(sum(magnetization[:,spin_group[i],1,1][:,1,:],dims=2)[:]/length(spin_group[i]))
Mz(i) = real.(sum(magnetization[:,spin_group[i],2,1][:,1,:],dims=2)[:]/length(spin_group[i]))

# Plot
p0 = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=["Mxy" "Mz" "Sequence"],
    shared_xaxes=true,
    vertical_spacing=0.1
)
for i=eachindex(spin_group)
    p1 = scatter(
        x=t, y=Mxy(i),
        name=labs[i],
        legendgroup=labs[i],
        marker_color=cols[i]
    )
    p2 = scatter(
        x=t,
        y=Mz(i),
        name=labs[i],
        legendgroup=labs[i],
        showlegend=false,
        marker_color=cols[i]
    )
    add_trace!(p0, p1, row=1, col=1)
    add_trace!(p0, p2, row=2, col=1)
end
relayout!(p0,
    yaxis_range=[0, 0.4],
    xaxis_range=[RR*dummy_heart_beats, RR*dummy_heart_beats+.250]
)
p0

# Simulating first heartbeat
FAs = 0:10:180          # flip angle [deg]
RRs = 60 ./ (40:10:100)  # RR [s]

mag1 = zeros(ComplexF64, im_segments, Niso*3, length(FAs), length(RRs))
#size(mag1)
@progress for (m, RR) = enumerate(RRs), (n, α) = enumerate(FAs)
#@progress for (m, RR) = enumerate(RRs)
    seq_params1 = merge(seq_params, (; im_flip_angle=[α, 80], RR))
    #seq_params1 = merge(seq_params, (; RR))
    #sim_params1 = merge(sim_params, Dict("sim_method"=>BlochDict()))
    sim_params1 = merge(sim_params, Dict("sim_method"=>BlochDict()))
    seq1        = BOOST(seq_params1...)
    obj1        = carotid_phantom(0)
    magaux = @suppress simulate(obj1, seq1, sys; sim_params=sim_params1)
    mag1[:, :, n, m] .= magaux[end-2im_segments+1:end-im_segments, :] # First heartbeat
end

# Labels
labels = ["Carotid", "Blood", "Fat (T₁=183 ms)"]
colors = ["blue", "red", "green", "purple"]
spins = [(1:Niso)', ((Niso + 1):(2Niso))', ((2Niso + 1):(3Niso))']
mean(x, dim) = sum(x; dims=dim) / size(x, dim)
std(x, dim; mu=mean(x, dim)) = sqrt.(sum(abs.(x .- mu) .^ 2; dims=dim) / (size(x, dim) - 1))

# Reducing tissues's signal
signal_1hb_caro = reshape(
    mean(abs.(mean(mag1[:, spins[1], :, :], 3)), 1), 1, length(FAs), length(RRs)
)
signal_1hb_bloo = reshape(
    mean(abs.(mean(mag1[:, spins[2], :, :], 3)), 1), 1, length(FAs), length(RRs)
)
signal_1hb_fat = reshape(
    mean(abs.(mean(mag1[:, spins[3], :, :], 3)), 1), 1, length(FAs), length(RRs)
)
signal_1hb_diff = abs.(signal_1hb_bloo .- signal_1hb_caro)

# Plotting results
# Mean
s11 = scatter(;
    x=FAs,
    y=signal_1hb_caro[:],
    name=labels[1],
    legendgroup=labels[1],
    line=attr(; color=colors[1]),
)
s12 = scatter(;
    x=FAs,
    y=signal_1hb_bloo[:],
    name=labels[2],
    legendgroup=labels[2],
    line=attr(; color=colors[2]),
)
s13 = scatter(;
    x=FAs,
    y=signal_1hb_diff[:],
    name="|Blood-Caro|",
    legendgroup="|Blood-Caro|",
    line=attr(color=colors[4])
)

#= s14 = scatter(;
    x=FAs,
    y=signal_1hb_fat[:],
	name=labels[3],
	legendgroup=labels[3],
	line=attr(color=colors[3])
) =#

# Plots
fig1 = plot([s11, s12, s13])
relayout!(
    fig1;
    title="First heartbeat",
    #title="First heartbeat (RR = $(round(Int, RRs[1] * 1000)) ms)",
    yaxis=attr(; title="Signal [a.u.]", tickmode="array"),
    xaxis=attr(;
        title="Flip Angle [deg]",
        tickmode="array",
        tickvals=[FAs[1], 80, 110, 130, 150,FAs[end]],
        constrain="domain",
    ),
    font=attr(; family="CMU Serif", size=16, scaleanchor="x", scaleratio=1),
    yaxis_range=[0, 0.3],
    xaxis_range=[FAs[1], FAs[end]],
    width=600,
    height=400,
    hovermode="x unified",
)
fig1
## Imagen simulacion 1HB

#Second heartbeat
FAs = 0:10:180          # flip angle [deg]
RRs = 60 ./ (40:10:100)  # RR [s]

mag2 = zeros(ComplexF64, im_segments, Niso*3, length(FAs), length(RRs))
@progress for (m, RR) = enumerate(RRs), (n, α ) = enumerate(FAs)
    seq_params2 = merge(seq_params, (; im_flip_angle=[110,α], RR))
    sim_params2 = merge(sim_params, Dict("sim_method"=>BlochDict()))
    seq2        = BOOST(seq_params2...)
    obj2        = carotid_phantom(0)
    magaux = @suppress simulate(obj2, seq2, sys; sim_params=sim_params2)
    mag2[:, :, n, m] .= magaux[end-im_segments+1:end, :] #Last heartbeat
end

## Calculating results
signal_2hb_caro = reshape(
    mean(abs.(mean(mag2[:, spins[1], :, :], 3)), 1), 1, length(FAs), length(RRs)
)
signal_2hb_bloo2 = reshape(
    mean(abs.(mean(mag2[:, spins[2], :, :], 3)), 1), 1, length(FAs), length(RRs)
)
signal_2hb_fat = reshape(
    mean(abs.(mean(mag2[:, spins[3], :, :], 3)), 1), 1, length(FAs), length(RRs)
)

signal_2hb_diff = abs.(signal_2hb_bloo2 .- signal_2hb_caro)

# Plotting results
# Mean
s21 = scatter(;
    x=FAs,
    y=signal_2hb_caro[:],
    name=labels[1],
    legendgroup=labels[1],
    line=attr(; color=colors[1]),
)
s22 = scatter(;
    x=FAs,
    y=signal_2hb_bloo2[:],
    name=labels[2],
    legendgroup=labels[2],
    line=attr(; color=colors[2]),
)
#= s23 = scatter(;
    x=FAs,
    y=signal_2hb_fat[:],
    name=labels[3],
    legendgroup=labels[3],
    line=attr(; color=colors[3]),
) =#
s24 = scatter(;
    #
    x=FAs,
    y=signal_2hb_diff[:],
    name="|Blood-Myoc|",
	legendgroup="|Blood-Myoc|",
	line=attr(color=colors[4])
)

# Plots
fig2 = plot([s21, s22, s24])
relayout!(
    fig2;
    title="Second heartbeat",
    yaxis=attr(; title="Signal [a.u.]", tickmode="array"),
    xaxis=attr(;
        title="Flip angle [deg]",
        tickmode="array",
        tickvals=[FAs[1], 80, 110, 130, FAs[end]],
        constrain="domain",
    ),
    font=attr(; family="CMU Serif", size=16, scaleanchor="x", scaleratio=1),
    yaxis_range=[0, 0.3],
    xaxis_range=[FAs[1], FAs[end]],
    width=600,
    height=400,
    hovermode="x unified",
)
fig2

#Fat Saturation
FFAs = 10:10:260          # flip angle [deg]
Δfs = (-1:0.2:1) .* (γ * sys.B0 * 1e-6)  # off-resonance Δf [s]
mag3 = zeros(ComplexF64, im_segments, Niso*3, length(FFAs), length(Δfs))
@progress for (m, Δf) = enumerate(Δfs), (n, FatSat_flip_angle) = enumerate(FFAs)
	seq_params3 = merge(seq_params, (; FatSat_flip_angle))
	sim_params3 = merge(sim_params, Dict("sim_method"=>BlochDict()))
	seq3        = BOOST(seq_params3...)
	obj3        = carotid_phantom(Δf)
	magaux = @suppress simulate(obj3, seq3, sys; sim_params=sim_params3)
	mag3[:, :, n, m] .= magaux[end-im_segments+1:end, :] # Last heartbeat
end

#Fat Saturation
signal_2hb_caro = reshape(
    mean(abs.(mean(mag3[:, spins[1], :, :], 3)), 1), length(FFAs), length(Δfs)
)
signal_2hb_bloo2 = reshape(
    mean(abs.(mean(mag3[:, spins[2], :, :], 3)), 1), length(FFAs), length(Δfs)
)
signal_2hb_fat = reshape(
    mean(abs.(mean(mag3[:, spins[3], :, :], 3)), 1), length(FFAs), length(Δfs)
)

# Plotting results
s14 = scatter(;
    x=FFAs,
    y=signal_2hb_caro[:],
    name=labels[1],
    legendgroup=labels[1],
    line=attr(; color=colors[1]),
)
s24 = scatter(;
    x=FFAs,
    y=signal_2hb_bloo2[:],
    name=labels[2],
    legendgroup=labels[2],
    line=attr(; color=colors[2]),
)
s34 = scatter(;
    x=FFAs,
    y=signal_2hb_fat[:],
    name=labels[3],
    legendgroup=labels[3],
    line=attr(; color=colors[3]),
)
# Plots
figFatSat = plot([s14, s24, s34])
relayout!(
    figFatSat;
    title="FatSat",
    yaxis=attr(; title="Signal [a.u.]", tickmode="array"),
    xaxis=attr(;
        title="FatSat flip angle [deg]",
        tickmode="array",
        tickvals=[FFAs[1], 130, 150, 180, FFAs[end]],
        #tickvals=[FAs[1], 100, 120, 160, FAs[end]],
        constrain="domain",
    ),
    font=attr(; family="CMU Serif", size=16, scaleanchor="x", scaleratio=1),
    yaxis_range=[0, 0.4],
    xaxis_range=[FFAs[1], FFAs[end]],
    width=600,
    height=400,
    hovermode="x unified",
)
figFatSat
# Save figures
savefig(figFatSat, "FatSat.html")
savefig(fig1, "bSSFP_1hb.html")
savefig(fig2, "bSSFP_2hb.html")

#Simulacion para iNav_lines=2
![Resultado iNav=2](simulaciones%20FAs/FAs_iNav_2.png)

#Simulacion para iNav_lines=4
![Resultado iNav=4](simulaciones%20FAs/FAs_iNavs_4.png)

#Simulacion para iNav_lines=6
![Resultado iNav=6](simulaciones%20FAs/FAs_iNavs_6.png)
