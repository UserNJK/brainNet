# -*- coding: utf-8 -*-
"""Generate the BioNeuron technical report PDF (reportlab / Platypus)."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, HRFlowable, ListFlowable,
                                ListItem)

OUT = "BioNeuron_Technical_Report.pdf"

INK   = colors.HexColor("#1a1a2e")
ACC   = colors.HexColor("#0b3d91")   # deep blue
GREY  = colors.HexColor("#555555")
LGREY = colors.HexColor("#efefef")
RULE  = colors.HexColor("#c9c9c9")

ss = getSampleStyleSheet()
def S(name, parent=None, **kw):
    return ParagraphStyle(name, parent=parent or ss["Normal"], **kw)

title_s  = S("t",  fontName="Helvetica-Bold", fontSize=23, leading=27, textColor=INK)
sub_s    = S("s",  fontName="Helvetica",      fontSize=12, leading=16, textColor=GREY)
h1_s     = S("h1", fontName="Helvetica-Bold", fontSize=14.5, leading=18, textColor=ACC,
             spaceBefore=15, spaceAfter=6, keepWithNext=1)
h2_s     = S("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=INK,
             spaceBefore=9, spaceAfter=3, keepWithNext=1)
body_s   = S("b",  fontName="Helvetica",      fontSize=9.6, leading=13.6, textColor=INK,
             alignment=TA_JUSTIFY, spaceAfter=5)
bul_s    = S("bl", parent=body_s, spaceAfter=2, leftIndent=6)
cap_s    = S("cap",fontName="Helvetica-Oblique", fontSize=8.3, leading=10.5, textColor=GREY,
             spaceAfter=8)
code_s   = S("c",  fontName="Courier", fontSize=8.2, leading=10.5, textColor=INK,
             backColor=LGREY, borderPadding=5, spaceAfter=6, spaceBefore=2)
ref_s    = S("r",  fontName="Helvetica", fontSize=8.6, leading=11.2, textColor=INK,
             leftIndent=14, firstLineIndent=-14, spaceAfter=3)

story = []
def h1(t): story.append(Paragraph(t, h1_s))
def h2(t): story.append(Paragraph(t, h2_s))
def p(t):  story.append(Paragraph(t, body_s))
def cap(t):story.append(Paragraph(t, cap_s))
def code(t): story.append(Paragraph(t.replace(" ", "&nbsp;").replace("\n","<br/>"), code_s))
def sp(h=6): story.append(Spacer(1, h))
def bullets(items):
    story.append(ListFlowable(
        [ListItem(Paragraph(i, bul_s), leftIndent=14, value="•") for i in items],
        bulletType="bullet", start="•", leftIndent=10))

def tbl(header, rows, widths, fs=8.4, align=None):
    data = [[Paragraph("<b>%s</b>" % c, S("th", fontName="Helvetica-Bold", fontSize=fs,
              leading=fs+2, textColor=colors.white)) for c in header]]
    for r in rows:
        data.append([Paragraph(str(c), S("td", fontName="Helvetica", fontSize=fs,
              leading=fs+2.4, textColor=INK)) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    stylecmds = [
        ("BACKGROUND",(0,0),(-1,0), ACC),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f4f6fb")]),
        ("GRID",(0,0),(-1,-1),0.4, RULE),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]
    if align:
        for col, a in align.items():
            stylecmds.append(("ALIGN",(col,0),(col,-1),a))
    t.setStyle(TableStyle(stylecmds))
    story.append(t); sp(8)

# ---------- footer / page numbers ----------
def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.4)
    canvas.line(72, 54, letter[0]-72, 54)
    canvas.setFont("Helvetica", 8); canvas.setFillColor(GREY)
    canvas.drawString(72, 42, "BioNeuron - Technical Report (bioneuron v1.8)")
    canvas.drawRightString(letter[0]-72, 42, "Page %d" % doc.page)
    canvas.restoreState()

# =====================================================================
#  TITLE
# =====================================================================
sp(40)
story.append(Paragraph("BioNeuron", title_s))
story.append(Paragraph("A Biologically Accurate Spiking Neural Network:<br/>"
                       "Design, Mechanisms, Simulations, and Results", sub_s))
sp(10)
story.append(HRFlowable(width="100%", thickness=1.1, color=ACC))
sp(10)
p("<b>Author:</b> Jugal Kishore Nagulapalli &nbsp;&nbsp;|&nbsp;&nbsp; "
  "<b>Software:</b> bioneuron v1.8 (MIT) &nbsp;&nbsp;|&nbsp;&nbsp; "
  "<b>Stack:</b> Python 3.10, PyTorch 2.6 (CUDA)")
sp(10)
p("<b>Summary.</b> BioNeuron is a from-scratch cortical spiking neural network in which every "
  "neuron follows biophysical rules. Learning is driven by local spike-timing plasticity and "
  "reward/stress neuromodulation, not by backpropagation. Connectivity is not fixed: synapses "
  "are grown from correlated activity, strengthened or weakened by plasticity, and pruned. The "
  "system reproduces a set of standard results from computational neuroscience (spike-timing "
  "learning, spike-frequency adaptation, short-term plasticity, gamma oscillations, "
  "self-organized criticality, dendritic nonlinearity) and learns simple tasks from its own "
  "self-generated feedback. This report states each mechanism, the equation it implements, the "
  "reference it follows, and the measured result. All numbers are from runs on the machine "
  "described in Section 9.")
sp(6)
p("<b>Design rule.</b> Biological accuracy is the priority. A correctness fix (for example, the "
  "spike-timing sign correction in Section 5.1) is applied unconditionally. Any mechanism that "
  "changes the dynamics of existing experiments is off by default and enabled by one flag, so "
  "earlier results stay reproducible.")

sp(14)
h2("Contents")
toc = [
 "1. Overview", "2. Neuron model", "3. Synapses and connectivity",
 "4. Network structure (cortical microcircuit)", "5. Plasticity and learning",
 "6. Neurochemistry", "7. Measurement and analysis tools",
 "8. Simulations and results", "9. Performance and hardware",
 "10. Correlates of awareness (scope and limits)",
 "11. Optimization roadmap", "12. Software and reproducibility", "13. References"]
for t in toc:
    story.append(Paragraph(t, S("toc", parent=body_s, spaceAfter=1.5, leftIndent=8)))

story.append(PageBreak())

# =====================================================================
# 1 OVERVIEW
# =====================================================================
h1("1. Overview")
p("A conventional artificial neural network is a set of continuous-valued units connected by a "
  "fixed weight matrix trained by gradient descent. BioNeuron is different on all three counts. "
  "Units are spiking neurons with a membrane voltage that integrates input and emits discrete "
  "spikes. Weights change by local rules that depend only on the relative timing of pre- and "
  "post-synaptic spikes and on broadcast chemical signals. The wiring itself is dynamic: it "
  "starts sparse and random, and is reshaped by activity.")
p("The model is organized as composable layers, each grounded in a specific piece of biology:")
bullets([
 "<b>Membrane dynamics</b> - leaky integrate-and-fire, optionally extended to the adaptive "
 "exponential (AdEx) model, with spike-frequency adaptation.",
 "<b>Receptors and channels</b> - fast (AMPA-like) and slow voltage-gated NMDA excitation, "
 "GABA inhibition, gap junctions, and a two-compartment active dendrite.",
 "<b>Cell types</b> - regular-spiking excitatory pyramidal cells and fast-spiking / "
 "low-threshold inhibitory interneurons, each with distinct intrinsic parameters.",
 "<b>Plasticity</b> - spike-timing-dependent plasticity (excitatory and inhibitory), "
 "short-term plasticity, homeostatic scaling, and three-factor reward/stress learning.",
 "<b>Structure</b> - Dale's law, an excitatory/inhibitory microcircuit, and optional columns.",
 "<b>Chemistry</b> - an extensible neurochemical layer (dopamine, serotonin, noradrenaline, "
 "acetylcholine, cortisol, and others) with release and clearance kinetics.",
 "<b>Instrumentation</b> - analyzers for oscillations, criticality, spike statistics, "
 "cell assemblies, and a battery of consciousness-correlate markers.",
])
p("Two connectivity backends are provided: a dense N-by-N weight matrix (simple, exact) and a "
  "sparse edge list whose cost scales with the number of synapses rather than N squared. The "
  "sparse backend reproduces the dense one exactly (Section 8.3).")

# =====================================================================
# 2 NEURON MODEL
# =====================================================================
h1("2. Neuron model")
h2("2.1 Membrane dynamics (leaky integrate-and-fire)")
p("Each neuron holds a membrane potential V (in mV). Between spikes it integrates leak, "
  "synaptic, adaptation, external, and noise currents:")
code("dV/dt = (V_rest - V)/tau_m  +  I_syn/C_m  -  w_adapt/C_m  +  I_ext  +  noise")
p("A spike is emitted when V crosses the threshold V_thresh; V is then reset to V_reset and the "
  "neuron is held refractory for t_ref. Default constants follow cortical values: "
  "V_rest = -70, V_thresh = -55, V_reset = -75 mV, tau_m = 20 ms, t_ref = 2 ms. Integration is "
  "explicit Euler with dt = 0.1 ms. Noise is scaled by dt by default; an option scales it by "
  "sqrt(dt), which is the correct scaling for a stochastic (Euler-Maruyama) update and lets "
  "modest noise drive realistic spontaneous firing (Section 8.2).")
h2("2.2 Spike-frequency adaptation")
p("Real neurons fire quickly then slow down: a calcium- and sodium-activated potassium current "
  "hyperpolarizes the cell after each spike. This is modelled by an adaptation variable w_adapt "
  "(Brette and Gerstner 2005) that decays toward a subthreshold drive and is incremented on "
  "each spike:")
code("dw_adapt/dt = (a*(V - V_rest) - w_adapt)/tau_w ;   on spike:  w_adapt += b")
p("It is on by default because it is nearly universal in cortex and stabilizes firing.")
h2("2.3 Adaptive exponential integrate-and-fire (AdEx)")
p("Optionally the hard threshold is replaced by a smooth, voltage-dependent spike-initiation "
  "term that reproduces a realistic upstroke and, together with the adaptation parameters, the "
  "main cortical firing patterns (tonic, adapting, bursting; Naud et al. 2008):")
code("I_exp = (Delta_T/tau_m) * exp((V - V_T)/Delta_T)")
p("The exponential is clamped at the peak voltage to prevent numerical overflow, and a spike is "
  "detected when V reaches the peak.")
h2("2.4 NMDA receptors")
p("Excitatory transmission has a fast component and a slow NMDA component whose conductance is "
  "gated by voltage through the magnesium block (Jahr and Stevens 1990). Near rest the channel "
  "is closed; depolarization opens it, so NMDA acts as a coincidence detector and supplies the "
  "slow recurrent drive needed to sustain persistent activity:")
code("B(V) = 1 / (1 + [Mg] * exp(-0.062*V) / 3.57)")
h2("2.5 Gap junctions and dendrites")
p("<b>Gap junctions</b> are electrical connections that pass current directly between coupled "
  "neurons; wired among fast-spiking interneurons they sharpen synchrony (Galarreta and Hestrin "
  "1999). <b>Dendrites:</b> a two-compartment option gives each neuron a separate dendritic "
  "compartment that receives excitation, integrates it, and fires regenerative dendritic spikes "
  "that then drive the soma (Poirazi and Mel 2003; Larkum 2013). This makes a single neuron a "
  "two-stage nonlinear unit (see the supralinear-summation result, Section 8.11).")
h2("2.6 Cell types")
p("Neuron parameters can be set per neuron. Three cortical types are provided:")
tbl(["Type", "Class", "tau_m (ms)", "t_ref (ms)", "Adaptation", "Role"],
    [["RS", "excitatory", "20", "2.0", "strong", "regular-spiking pyramidal"],
     ["FS", "inhibitory", "10", "1.0", "none", "fast-spiking PV+ interneuron"],
     ["LTS","inhibitory", "15", "2.0", "weak", "low-threshold SST+ interneuron"]],
    [0.7*inch, 1.0*inch, 0.9*inch, 0.9*inch, 0.9*inch, 2.0*inch])

# =====================================================================
# 3 SYNAPSES
# =====================================================================
h1("3. Synapses and connectivity")
h2("3.1 Transmission")
p("A spike from neuron i adds current to each target j, scaled by the weight W[i,j] and the "
  "sign of neuron i (Dale's law: excitatory +1, inhibitory -1). Two forms are available: "
  "current-based (net drive is g_e - g_i) and conductance-based, which uses reversal potentials "
  "so inhibition is divisive (shunting):")
code("I_syn = g_e*(E_exc - V) + g_i*(E_inh - V)      [conductance-based]")
p("Conductances decay with separate excitatory and inhibitory time constants. Axonal conduction "
  "delays (finite propagation time) are optional and drawn per axon.")
h2("3.2 Backends")
p("The dense backend stores W as an N-by-N matrix; propagation is a matrix-vector product. The "
  "sparse backend stores an edge list (pre, post, weight); propagation is a scatter-add over "
  "edges, with cost proportional to the number of synapses. The sparse backend is exact with "
  "respect to the dense one (Section 8.3) and removes the N-squared memory ceiling.")
h2("3.3 Growth and pruning")
p("Connectivity self-organizes. A growth cone periodically computes pairwise correlations of "
  "recent spike trains and forms synapses between neurons whose activity is correlated above a "
  "threshold, modelling activity-dependent axon guidance. Weak synapses are pruned. This is the "
  "mechanism by which structure emerges from dynamics rather than being imposed.")

# =====================================================================
# 4 NETWORK STRUCTURE
# =====================================================================
h1("4. Network structure (cortical microcircuit)")
p("Beyond a homogeneous random network, a structured cortex can be built: the canonical "
  "excitatory/inhibitory microcircuit with distinct connection probabilities and strengths for "
  "the four pathways (E to E, E to I, I to E, I to I), optionally partitioned into columns with "
  "dense intra-column and sparse inter-column wiring. Fast-spiking interneurons providing fast "
  "feedback inhibition to excitatory cells generate gamma-band oscillations by the "
  "pyramidal-interneuron (PING) mechanism (Cardin et al. 2009; Buzsaki and Wang 2012). This is "
  "the first rhythm that a homogeneous network cannot produce; it requires the distinct, fast, "
  "non-adapting interneuron class wired into feedback inhibition (result in Section 8.10).")

# =====================================================================
# 5 PLASTICITY
# =====================================================================
h1("5. Plasticity and learning")
h2("5.1 Spike-timing-dependent plasticity (STDP)")
p("The core learning rule follows Bi and Poo (1998), implemented as an online pair rule "
  "(Morrison et al. 2008). Each neuron keeps a decaying trace of its spikes. When the "
  "post-synaptic neuron fires and the pre-synaptic neuron fired shortly before "
  "(pre-before-post), the synapse is strengthened (LTP); the reverse order weakens it (LTD):")
code("post fires:  W[pre,post] += A_plus  * trace_pre[pre]      (LTP)\n"
     "pre  fires:  W[pre,post] -= A_minus * trace_post[post]    (LTD)")
p("<b>Correctness note.</b> An earlier version had this rule sign-inverted: pre-before-post "
  "depressed the synapse, making learning anti-Hebbian and contradicting the cited rule. A "
  "direct two-neuron test confirmed the inversion (a pre-then-post pair moved the weight from "
  "0.500 to 0.489, i.e. depression where potentiation was required). The corrected rule "
  "potentiates pre-before-post and depresses post-before-pre, with traces registered after the "
  "weight change (correct pair ordering). This changes any result that depends on plasticity.")
h2("5.2 Inhibitory STDP")
p("Inhibitory synapses follow a separate rule (Vogels et al. 2011) that grows inhibition onto "
  "over-active cells and shrinks it onto quiet ones, driving each cell toward a target rate. "
  "This establishes detailed excitatory/inhibitory balance and counteracts runaway excitation.")
h2("5.3 Short-term plasticity")
p("Synaptic strength also changes on the 10-to-100 ms scale from vesicle depletion (depression) "
  "and residual calcium (facilitation), following Tsodyks and Markram (1997). Per axon, a "
  "utilization variable u facilitates and a resource variable x depletes on each spike; the "
  "released efficacy is u*x. Facilitation is the mechanism for working memory held in synapses "
  "rather than in persistent firing (Mongillo et al. 2008).")
h2("5.4 Homeostatic scaling")
p("Slow multiplicative rescaling of a neuron's incoming weights toward a target firing rate "
  "(Turrigiano and Nelson 2004) keeps STDP from collapsing the network into silence or "
  "saturation.")
h2("5.5 Three-factor reward and stress learning")
p("For reinforcement, STDP does not edit weights directly; it lays down a slow eligibility "
  "trace that tags recently active (causal) synapses. A later neuromodulator converts the tag "
  "into a weight change (Fremaux and Gerstner 2016; Izhikevich 2007):")
code("dW = lr * M * eligibility ;   M = da_gain*(dopamine - 1) - cort_gain*cortisol")
p("Dopamine (reward) potentiates the tagged pathway; cortisol (a slow stress hormone, Joels and "
  "Krugers 2007) depresses it and, when sustained, impairs plasticity. The API is "
  "reward(), punish(), and feedback(correct). One subtlety was found and fixed: the eligibility "
  "tag must be the causal (LTP-shaped) component, not the net STDP change, because with "
  "A_minus over A_plus the net tag is negative under synchronous activity, which would invert "
  "the sign of reward and punishment.")

# =====================================================================
# 6 NEUROCHEMISTRY
# =====================================================================
h1("6. Neurochemistry")
p("Neuromodulation is provided by an extensible chemical layer. Each chemical is an object with "
  "a baseline level, a clearance time constant, and a set of effects on five control axes that "
  "the simulator reads (dopamine, serotonin, noradrenaline, acetylcholine, cortisol). Because a "
  "chemical influences the network only by pushing these axes, the layer is a drop-in "
  "replacement for the simple five-scalar modulator, and new chemicals affect the dynamics "
  "without changing the core. The default set models the five primaries plus histamine, "
  "adenosine (sleep pressure), melatonin, oxytocin, and endorphin, with fast clearance for "
  "dopamine and slow clearance for the hormones. Verified behaviour is in Section 8.9.")

story.append(PageBreak())

# =====================================================================
# 7 MEASUREMENT TOOLS
# =====================================================================
h1("7. Measurement and analysis tools")
p("The following analyzers read activity without changing it. They exist to test whether the "
  "network is in a biologically realistic regime.")
bullets([
 "<b>Oscillation analyzer</b> - treats the population firing rate as a local-field-potential "
 "proxy and reports power in the delta, theta, alpha, beta, and gamma bands, the dominant band, "
 "and the peak frequency (Buzsaki and Draguhn 2004).",
 "<b>Criticality analyzer</b> - the branching ratio (average number of descendant spikes per "
 "spike; near 1 at criticality) and neuronal-avalanche statistics (Beggs and Plenz 2003).",
 "<b>Spike statistics</b> - inter-spike-interval irregularity (coefficient of variation; near 1 "
 "is Poisson-like) and the Fano factor, indicating the asynchronous-irregular cortical regime "
 "(Softky and Koch 1993; Brunel 2000).",
 "<b>Cell-assembly detection</b> - groups of synchronously firing, mutually connected neurons "
 "read out as emergent assemblies (Hebb 1949).",
])
p("A separate battery estimates operational correlates of awareness (Section 10).")

# =====================================================================
# 8 RESULTS
# =====================================================================
h1("8. Simulations and results")
p("Each mechanism was checked against a known-correct expectation before being kept. Fast "
  "checks run as an automated test suite (9 of 9 passing). Longer learning runs are reproduced "
  "by demonstration functions. The tables below give the measured numbers.")

h2("8.1 Spike-timing plasticity (direction)")
tbl(["Test", "Expectation", "Measured"],
    [["Pre-before-post pair", "potentiate (LTP)", "dW = +0.0095"],
     ["Post-before-pre pair", "depress (LTD)", "dW = -0.0114"]],
    [2.2*inch, 2.2*inch, 2.0*inch])
cap("Confirms the corrected causal direction. The earlier inverted rule produced the opposite signs.")

h2("8.2 Single-neuron biophysics")
tbl(["Mechanism", "Expectation", "Measured"],
    [["Spike-freq. adaptation", "fewer spikes vs off", "1318 -> 1189 spikes"],
     ["AdEx", "fires, numerically stable, adapts", "stable; more adaptation -> fewer spikes"],
     ["NMDA gate B(V)", "closed at rest, open when depolarized", "B(-70 mV)=0.044, B(0 mV)=0.781"],
     ["sqrt(dt) noise", "modest noise drives spontaneous firing", "silent with dt-scaling; fires with sqrt(dt)"]],
    [1.8*inch, 2.3*inch, 2.3*inch])

h2("8.3 Sparse backend equivalence and scale")
tbl(["Check", "Result"],
    [["Propagation, dense vs sparse", "match to 5e-7 (floating-point reduction order)"],
     ["STDP per-edge weight update", "exact, 0.0 difference"],
     ["Spike trains, matched wiring", "100% identical over 150 steps"],
     ["Memory at N=3000 (439k edges)", "7 MB (sparse) vs 72 MB (dense); grows as N^2 for dense"]],
    [2.7*inch, 3.7*inch])

h2("8.4 Short-term plasticity")
tbl(["Regime", "Expectation", "Measured across a spike train"],
    [["Depression", "efficacy falls", "0.75 -> 0.013"],
     ["Facilitation", "utilization u rises", "0.28 -> 0.67"]],
    [1.6*inch, 1.9*inch, 2.9*inch])

h2("8.5 Network regime diagnostics")
tbl(["Diagnostic", "Expectation", "Measured"],
    [["Branching ratio (constant input)", "1.0 (critical)", "1.000"],
     ["Branching ratio (0.8 geometric decay)", "0.8", "0.798"],
     ["ISI CV (clockwork firing)", "near 0", "0.00"],
     ["ISI CV (Poisson firing)", "near 1", "0.88"],
     ["Oscillation band powers", "sum to about 1 in range", "sum = 0.99"]],
    [2.6*inch, 1.9*inch, 1.9*inch])

h2("8.6 Reward and stress learning (three-factor)")
p("An input group A was stimulated repeatedly and then given a teaching signal. Weight is the "
  "mean of the A-to-A synapses; identical on the dense and sparse backends.")
tbl(["Condition", "A-to-A weight"],
    [["No teaching signal", "0.049 -> 0.049 (no drift; learning needs a signal)"],
     ["Reward (dopamine)", "0.049 -> 0.295 (potentiated)"],
     ["Punishment (cortisol)", "0.049 -> 0.000 (depressed)"],
     ["Idle group during reward", "0.048 -> 0.048 (untouched; credit is specific)"]],
    [1.9*inch, 4.5*inch])

h2("8.7 Neurochemistry")
tbl(["Test", "Measured"],
    [["Histamine raises the arousal axis", "1.00 -> 1.30"],
     ["Adenosine lowers arousal and attention", "both fall below 0.95"],
     ["Oxytocin buffers cortisol", "cortisol axis reduced"],
     ["Clearance kinetics", "dopamine fast (2.0->1.08 in 500 ms); cortisol slow (2.0->1.87 in 200 ms)"]],
    [2.7*inch, 3.7*inch])

h2("8.8 Closed-loop learning from self-generated feedback")
p("The network's own decoded output decides whether it was right, and reward or stress is "
  "applied automatically. Operant conditioning (rewarding the network for producing a target "
  "output on cue; Fetz 1969) learns reliably. Two-way discrimination learns once three "
  "ingredients are combined: winner-take-all competition between output groups, a dopamine "
  "reward-prediction-error baseline (Schultz et al. 1997), and a dedicated plastic input-to-"
  "output pathway (R-STDP; Izhikevich 2007; Legenstein et al. 2008).")
tbl(["Task", "Result"],
    [["Operant conditioning", "response rate 0.27 -> 1.00 (learns to respond on cue)"],
     ["Two-way discrimination", "chance -> about 0.85 accuracy mid-run, then a late instability"]],
    [2.2*inch, 4.2*inch])
cap("The late instability is unregulated R-STDP (positive feedback and weight saturation); "
    "inhibitory STDP or homeostatic scaling are the intended stabilizers. With one of the three "
    "ingredients removed, discrimination stays at chance.")

h2("8.9 Gamma oscillations (structured cortex, PING)")
tbl(["Measurement", "Result (N = 800, excitatory cells driven)"],
    [["Dominant band", "gamma (relative power 0.53); rhythm at about 24-30 Hz"],
     ["Fast-spiking interneuron rate", "160 Hz (characteristic PV+ signature)"],
     ["Regular-spiking pyramidal rate", "54 Hz"]],
    [2.2*inch, 4.2*inch])

h2("8.10 Dendritic computation (supralinear summation)")
p("Two input groups each drove a target neuron; the somatic response was measured to each group "
  "alone and to both together.")
tbl(["Neuron", "Group A", "Group B", "A + B (linear)", "A and B (measured)"],
    [["Active dendrite", "4 spikes", "4 spikes", "8 expected", "10 (supralinear, ratio 1.25)"],
     ["Point neuron", "0", "0", "0", "0 (cannot detect the input)"]],
    [1.4*inch, 0.85*inch, 0.85*inch, 1.1*inch, 2.2*inch])
cap("The active dendrite sums supralinearly (the dendritic-spike signature); the point neuron is "
    "silent to the same distributed weak input.")

# =====================================================================
# 9 PERFORMANCE
# =====================================================================
h1("9. Performance and hardware")
p("Throughput was measured with the sparse backend at dt = 0.1 ms. Real time means 10,000 steps "
  "per second. Hardware: a laptop CPU and an NVIDIA RTX 3070 Laptop GPU (8.6 GB).")
tbl(["Device", "Neurons N", "Synapses", "Steps/s", "Fraction of real time"],
    [["CPU", "1,000", "49k", "261", "2.6%"],
     ["CPU", "3,000", "439k", "29", "0.29%"],
     ["CPU", "8,000", "3.1M", "5.7", "0.06%"],
     ["RTX 3070", "3,000", "439k", "73", "0.73%"],
     ["RTX 3070", "10,000", "4.9M", "7.5", "0.08%"],
     ["RTX 3070", "30,000", "43.9M", "0.7", "0.01%"]],
    [1.1*inch, 1.1*inch, 1.0*inch, 1.0*inch, 1.9*inch],
    align={2:"RIGHT",3:"RIGHT",4:"RIGHT"})
p("The limit is speed, not memory, and the cause is the per-step Python loop and two "
  "O(N-squared) analyzers (the growth cone and assembly detection), which leave the GPU mostly "
  "idle. Memory (sparse backend) is about 24 bytes per synapse. At a biological 1,000 synapses "
  "per neuron this is about 24 KB per neuron:")
tbl(["Neurons", "Synapses", "Memory", "Fits on"],
    [["100,000", "1e8", "2.4 GB", "RTX 3070 (8 GB)"],
     ["1,000,000 (insect scale)", "1e9", "24 GB", "RTX 4090 (24 GB) / A100"],
     ["10,000,000", "1e10", "240 GB", "multi-GPU node"]],
    [2.0*inch, 1.1*inch, 1.1*inch, 2.2*inch])
p("On the RTX 3070, the practical range for the science (marker experiments, closed-loop "
  "learning, oscillations, criticality) is roughly 1,000 to 10,000 neurons. Reaching "
  "insect-scale networks requires both a 24 GB or larger GPU and a rewrite of the per-step core "
  "into fused GPU kernels; the current Python step loop, not the card, is the binding "
  "constraint.")

# =====================================================================
# 10 AWARENESS
# =====================================================================
h1("10. Correlates of awareness (scope and limits)")
p("A separate battery estimates operational markers that consciousness science uses to "
  "distinguish conscious from unconscious brain states, and validates them by requiring an "
  "awake, integrated network to separate from unconscious control conditions (anaesthetized, "
  "disconnected, structure-shuffled). This is an explicit and important limit: no measurement "
  "proves subjective experience. A high score means the network shows the dynamical signatures "
  "that, in real brains, accompany conscious states; it does not mean the network feels "
  "anything.")
bullets([
 "<b>Global ignition</b> - a strong stimulus triggers sustained, network-wide activity while a "
 "weak one stays local (global-workspace theory; Dehaene and Changeux 2011).",
 "<b>Perturbational complexity</b> - the spatiotemporal complexity of the response to a brief "
 "perturbation, using Lempel-Ziv complexity (Casali et al. 2013).",
 "<b>Information integration</b> - lagged mutual information between network halves (a tractable "
 "proxy, not the full measure).",
 "<b>Predictive mismatch</b> - habituation to a repeated stimulus and a rebound to a novel one.",
])
p("Validation to date: the awake network ignites (post-stimulus sustained activity 0.023) while "
  "the disconnected control does not (0.000), and complexity ordering is correct. On the "
  "question of hardware for a minimal ('bare') awareness: the honest biological reference point "
  "is insect scale, on the order of one million neurons (Barron and Klein 2016). That is a "
  "24 GB memory target and, more importantly, requires the compute rewrite above; scale is a "
  "necessary but not sufficient condition, and none of it settles whether anything is felt.")

# =====================================================================
# 11 OPTIMIZATION ROADMAP
# =====================================================================
h1("11. Optimization roadmap")
p("The Dragon Hatchling architecture (BDH; Kosowski et al. 2025) is a GPU-efficient, trainable "
  "model built from the same ingredients as BioNeuron (Hebbian fast weights, sparse positive "
  "activity, excitatory/inhibitory circuits). Its central efficiency trick - never form the "
  "N-by-N synaptic operator, factor it through a low-rank bottleneck - does not transfer to "
  "BioNeuron's synapses, which are structured-sparse, signed, thresholded, and NMDA-nonlinear; "
  "forcing that factorization would replace the model rather than optimize it. What does "
  "transfer is narrower and lives in the analysis and execution layers, not the biophysics:")
bullets([
 "<b>Active-set assembly detection</b> - the co-firing matrix is a correlation of spike rows, "
 "and a silent neuron contributes exactly zero, so restricting the computation to neurons that "
 "fired is bit-exact and removes the dominant cost past a few thousand neurons.",
 "<b>Sync-free stepping</b> - deferring the per-step device-to-host reads (about 5 per step) "
 "into once-per-chunk reads is bit-identical and gives roughly 1.5 to 3x at moderate N.",
 "<b>Optional low-rank sketch</b> of the correlation matrices removes the N-by-N allocation that "
 "limits the 8 GB GPU; it is approximate, so it is opt-in.",
])
p("Not advisable: making the synapses themselves globally low-rank (it changes the dynamics), "
  "or a factored eligibility trace on the dense backend (it is more compute, not less). A "
  "differentiable, backprop-trainable readout is possible but is a separate machine-learning "
  "capability, not a fidelity feature, and would be an opt-in module.")

# =====================================================================
# 12 SOFTWARE
# =====================================================================
h1("12. Software and reproducibility")
p("The project is a pip-installable package, <b>bioneuron</b> (v1.8, MIT). Modules: core "
  "(neurons, synapses, plasticity, growth, analyzers, orchestration); sparse_synapses "
  "(edge-list backend); neurochemistry (chemical layer); cortex (cell types and microcircuit); "
  "closed_loop (self-feedback tasks). A fast test suite (9 of 9 passing) covers the "
  "spike-timing direction, sparse/dense equivalence, adaptation, inhibitory STDP, neurochemistry "
  "axes, reward and stress learning, cell-type firing, and dendritic supralinearity. Every "
  "result in Section 8 is reproduced by these tests or by the demonstration functions.")

story.append(PageBreak())

# =====================================================================
# 13 REFERENCES
# =====================================================================
h1("13. References")
refs = [
 "Barron, A. B., and Klein, C. (2016). What insects can tell us about the origins of "
 "consciousness. PNAS 113(18), 4900-4908.",
 "Beggs, J. M., and Plenz, D. (2003). Neuronal avalanches in neocortical circuits. "
 "Journal of Neuroscience 23(35), 11167-11177.",
 "Bi, G. Q., and Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: "
 "dependence on spike timing, synaptic strength, and postsynaptic cell type. "
 "Journal of Neuroscience 18(24), 10464-10472.",
 "Brette, R., and Gerstner, W. (2005). Adaptive exponential integrate-and-fire model as an "
 "effective description of neuronal activity. Journal of Neurophysiology 94(5), 3637-3642.",
 "Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and inhibitory "
 "spiking neurons. Journal of Computational Neuroscience 8(3), 183-208.",
 "Buzsaki, G., and Draguhn, A. (2004). Neuronal oscillations in cortical networks. "
 "Science 304(5679), 1926-1929.",
 "Buzsaki, G., and Wang, X. J. (2012). Mechanisms of gamma oscillations. "
 "Annual Review of Neuroscience 35, 203-225.",
 "Cardin, J. A., et al. (2009). Driving fast-spiking cells induces gamma rhythm and controls "
 "sensory responses. Nature 459, 663-667.",
 "Casali, A. G., et al. (2013). A theoretically based index of consciousness independent of "
 "sensory processing and behavior. Science Translational Medicine 5(198), 198ra105.",
 "Dale, H. (1935). Pharmacology and nerve-endings. Proceedings of the Royal Society of Medicine "
 "28(3), 319-332.",
 "Dehaene, S., and Changeux, J. P. (2011). Experimental and theoretical approaches to conscious "
 "processing. Neuron 70(2), 200-227.",
 "Fetz, E. E. (1969). Operant conditioning of cortical unit activity. Science 163(3870), "
 "955-958.",
 "Fremaux, N., and Gerstner, W. (2016). Neuromodulated spike-timing-dependent plasticity, and "
 "theory of three-factor learning rules. Frontiers in Neural Circuits 9, 85.",
 "Galarreta, M., and Hestrin, S. (1999). A network of fast-spiking cells in the neocortex "
 "connected by electrical synapses. Nature 402, 72-75.",
 "Hebb, D. O. (1949). The Organization of Behavior. Wiley, New York.",
 "Hodgkin, A. L., and Huxley, A. F. (1952). A quantitative description of membrane current and "
 "its application to conduction and excitation in nerve. Journal of Physiology 117(4), 500-544.",
 "Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and "
 "dopamine signaling. Cerebral Cortex 17(10), 2443-2452.",
 "Jahr, C. E., and Stevens, C. F. (1990). Voltage dependence of NMDA-activated macroscopic "
 "conductances predicted by single-channel kinetics. Journal of Neuroscience 10(9), 3178-3182.",
 "Joels, M., and Krugers, H. J. (2007). LTP after stress: up or down? Neural Plasticity 2007, "
 "93202.",
 "Kosowski, A., Uznanski, P., Chorowski, J., Stamirowska, Z., and Bartoszkiewicz, M. (2025). "
 "The Dragon Hatchling: the missing link between the Transformer and models of the brain. "
 "arXiv:2509.26507.",
 "Larkum, M. (2013). A cellular mechanism for cortical associations: an organizing principle for "
 "the cerebral cortex. Trends in Neurosciences 36(3), 141-151.",
 "Legenstein, R., Pecevski, D., and Maass, W. (2008). A learning theory for reward-modulated "
 "spike-timing-dependent plasticity with application to biofeedback. PLoS Computational Biology "
 "4(10), e1000180.",
 "Mongillo, G., Barak, O., and Tsodyks, M. (2008). Synaptic theory of working memory. "
 "Science 319(5869), 1543-1546.",
 "Morrison, A., Diesmann, M., and Gerstner, W. (2008). Phenomenological models of synaptic "
 "plasticity based on spike timing. Biological Cybernetics 98(6), 459-478.",
 "Naud, R., Marcille, N., Clopath, C., and Gerstner, W. (2008). Firing patterns in the adaptive "
 "exponential integrate-and-fire model. Biological Cybernetics 99(4-5), 335-347.",
 "Poirazi, P., and Mel, B. W. (2003). Pyramidal neuron as two-layer neural network. "
 "Neuron 37(6), 989-999. (with Polsky, A., Mel, B. W., and Schiller, J. (2004), "
 "Nature Neuroscience 7, 621-627, on dendritic compartmental subunits.)",
 "Porkka-Heiskanen, T., et al. (1997). Adenosine: a mediator of the sleep-inducing effects of "
 "prolonged wakefulness. Science 276(5316), 1265-1268.",
 "Saper, C. B., Scammell, T. E., and Lu, J. (2005). Hypothalamic regulation of sleep and "
 "circadian rhythms. Nature 437, 1257-1263.",
 "Schultz, W., Dayan, P., and Montague, P. R. (1997). A neural substrate of prediction and "
 "reward. Science 275(5306), 1593-1599.",
 "Softky, W. R., and Koch, C. (1993). The highly irregular firing of cortical cells is "
 "inconsistent with temporal integration of random EPSPs. Journal of Neuroscience 13(1), "
 "334-350.",
 "Tsodyks, M. V., and Markram, H. (1997). The neural code between neocortical pyramidal neurons "
 "depends on neurotransmitter release probability. PNAS 94(2), 719-723.",
 "Turrigiano, G. G., and Nelson, S. B. (2004). Homeostatic plasticity in the developing nervous "
 "system. Nature Reviews Neuroscience 5(2), 97-107.",
 "Vogels, T. P., Sprekeler, H., Zenke, F., Clopath, C., and Gerstner, W. (2011). Inhibitory "
 "plasticity balances excitation and inhibition in sensory pathways and memory networks. "
 "Science 334(6062), 1569-1573.",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph("%d. %s" % (i, r), ref_s))

doc = SimpleDocTemplate(OUT, pagesize=letter,
                        leftMargin=72, rightMargin=72, topMargin=64, bottomMargin=64,
                        title="BioNeuron Technical Report", author="Jugal Kishore Nagulapalli")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("WROTE", OUT)
