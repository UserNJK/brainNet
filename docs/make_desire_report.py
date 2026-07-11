# -*- coding: utf-8 -*-
"""
make_desire_report.py — build the Desire-battery study PDF (reportlab / Platypus).
Reads docs/desire_study.json + docs/desire_figs/*.png produced by
src/run_desire_study.py, and writes docs/Desire_Study_Report.pdf.
"""
import os, json
# This OpenSSL build's md5 rejects the usedforsecurity kwarg that reportlab 4.x
# passes; shim it BEFORE reportlab imports md5 (its `from hashlib import md5`
# captures this wrapper).
import hashlib as _hashlib
_orig_md5 = _hashlib.md5
def _md5_compat(*a, **k):
    k.pop('usedforsecurity', None)
    return _orig_md5(*a, **k)
_hashlib.md5 = _md5_compat
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, HRFlowable, ListFlowable,
                                ListItem, Image)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'desire_figs')
OUT  = os.path.join(HERE, 'Desire_Study_Report.pdf')
DATA = json.load(open(os.path.join(HERE, 'desire_study.json')))

INK   = colors.HexColor("#1a1a2e")
ACC   = colors.HexColor("#0b3d91")
GREY  = colors.HexColor("#555555")
LGREY = colors.HexColor("#efefef")
RULE  = colors.HexColor("#c9c9c9")

ss = getSampleStyleSheet()
def S(name, parent=None, **kw):
    return ParagraphStyle(name, parent=parent or ss["Normal"], **kw)

title_s = S("t",  fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=INK)
sub_s   = S("s",  fontName="Helvetica",      fontSize=11.5, leading=15, textColor=GREY)
h1_s    = S("h1", fontName="Helvetica-Bold", fontSize=14.5, leading=18, textColor=ACC,
            spaceBefore=15, spaceAfter=6, keepWithNext=1)
h2_s    = S("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=INK,
            spaceBefore=9, spaceAfter=3, keepWithNext=1)
body_s  = S("b",  fontName="Helvetica",      fontSize=9.6, leading=13.8, textColor=INK,
            alignment=TA_JUSTIFY, spaceAfter=5)
bul_s   = S("bl", parent=body_s, spaceAfter=2, leftIndent=6)
cap_s   = S("cap",fontName="Helvetica-Oblique", fontSize=8.3, leading=10.5, textColor=GREY,
            spaceAfter=10, spaceBefore=2)
code_s  = S("c",  fontName="Courier", fontSize=8.2, leading=10.5, textColor=INK,
            backColor=LGREY, borderPadding=5, spaceAfter=6, spaceBefore=2)
ref_s   = S("r",  fontName="Helvetica", fontSize=8.6, leading=11.2, textColor=INK,
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
def fig(name, width=6.2*inch):
    path = os.path.join(FIGS, name)
    img = Image(path); ar = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width; img.drawHeight = width * ar
    story.append(img)

def tbl(header, rows, widths, fs=8.6, align=None):
    data = [[Paragraph("<b>%s</b>" % c, S("th", fontName="Helvetica-Bold", fontSize=fs,
              leading=fs+2, textColor=colors.white)) for c in header]]
    for r in rows:
        data.append([Paragraph(str(c), S("td", fontName="Helvetica", fontSize=fs,
              leading=fs+2.4, textColor=INK)) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    cmds = [("BACKGROUND",(0,0),(-1,0), ACC),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f4f6fb")]),
            ("GRID",(0,0),(-1,-1),0.4, RULE), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
            ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]
    if align:
        for col, a in align.items(): cmds.append(("ALIGN",(col,0),(col,-1),a))
    t.setStyle(TableStyle(cmds)); story.append(t); sp(8)

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.4)
    canvas.line(72, 54, letter[0]-72, 54)
    canvas.setFont("Helvetica", 8); canvas.setFillColor(GREY)
    canvas.drawString(72, 42, "The Desire Battery - an operational-marker study (bioneuron)")
    canvas.drawRightString(letter[0]-72, 42, "Page %d" % doc.page)
    canvas.restoreState()

# convenience accessors into the aggregated data ----------------------
AGG = DATA['agg']; CONDS = DATA['conditions']; MARKS = DATA['markers']
MLAB = DATA['marker_labels']
def mm(c, m): v = AGG[c]['markers'][m]; return v[0], v[1]
def di(c):    v = AGG[c]['desire_index']; return v[0], v[1]
def ph(c, p): v = AGG[c]['phases'][p]; return v[0], v[1]
def fmt(mean, sd, nd=2): return f"{mean:.{nd}f} &plusmn; {sd:.{nd}f}"

# =====================================================================
#  TITLE
# =====================================================================
sp(34)
story.append(Paragraph("The Desire Battery", title_s))
story.append(Paragraph("Operational Markers of Incentive Motivation in a "
                       "Biologically Accurate Spiking Neural Network", sub_s))
sp(9); story.append(HRFlowable(width="100%", thickness=1.1, color=ACC)); sp(9)
p("<b>Author:</b> Jugal Kishore Nagulapalli &nbsp;|&nbsp; <b>Software:</b> bioneuron "
  "(MIT) &nbsp;|&nbsp; <b>Stack:</b> Python 3.10, PyTorch (CUDA) &nbsp;|&nbsp; "
  f"<b>Runs:</b> N = {DATA['N']} neurons, {len(DATA['seeds'])} random seeds")
sp(8)
p("<b>Abstract.</b> Desire &mdash; the motivational pull toward a reward-predicting "
  "cue &mdash; is, in the incentive-salience framework of Berridge and Robinson, a "
  "dopamine-dependent process (&lsquo;wanting&rsquo;) separable from the hedonic "
  "impact of the reward itself (&lsquo;liking&rsquo;). We ask whether BioNeuron, a "
  "from-scratch cortical spiking network with dopamine-gated three-factor plasticity, "
  "exhibits the <i>operational markers</i> of wanting, and we validate those markers "
  "the only honest way &mdash; by dissociation from control conditions in which the "
  "machinery of wanting is disabled. Across five random seeds, an intact, "
  "dopamine-competent, contingently-trained network acquires cue-evoked pursuit, "
  "sustains it, loses it under extinction, revives it on reinstatement, and directs "
  "synaptic credit preferentially to the pathway that earned reward. A "
  "dopamine-lesioned control and a reward-decoupled (yoked) control show none of "
  "this. The intact network separates cleanly on the aggregate desire index "
  f"({di('INTACT')[0]:+.2f} vs {di('DA-LESION')[0]:+.2f} and {di('DECOUPLED')[0]:+.2f}). "
  "As with any such measurement, this is evidence of desire-like <i>dynamics</i>, "
  "not proof that anything is wanted.")
sp(6)
p("<b>Design rule (inherited).</b> Biological accuracy is the priority. The battery "
  "adds no new mechanism to the simulator; it only drives, probes, and scores the "
  "existing dopamine-gated learning path, exactly as the awareness battery does for "
  "consciousness correlates.")

# =====================================================================
# 1 THESIS
# =====================================================================
h1("1. Thesis")
p("<b>Central claim.</b> If &lsquo;desire&rsquo; is operationalised as incentive "
  "motivation &mdash; goal-directed pursuit of a reward-predicting cue, driven by a "
  "dopamine signal and directed at the specific object that produced reward &mdash; "
  "then BioNeuron exhibits it as an <i>emergent, measurable, and dissociable</i> "
  "property of its dopamine-gated three-factor plasticity, and does so only when that "
  "machinery is intact and the reward is contingent on the network&rsquo;s own action.")
p("This claim is deliberately narrow. It is a claim about <i>function</i> "
  "(the network behaves as a system that wants), not about <i>phenomenology</i> "
  "(that it feels wanting). The distinction is not a hedge: it is the same "
  "epistemic limit that the consciousness literature places on every physiological "
  "marker, and Section 7 states it plainly.")
p("<b>Hypotheses.</b>")
bullets([
 "<b>H1 (acquisition).</b> Cue-evoked pursuit rises across rewarded training in the "
 "intact network and not in the controls (incentive salience is <i>learned</i>).",
 "<b>H2 (dopamine-dependence).</b> Removing dopamine&rsquo;s ability to gate "
 "plasticity (da_gain = 0) abolishes acquisition while leaving spiking intact "
 "&mdash; the network still acts, but no longer <i>wants</i> (Berridge 2007).",
 "<b>H3 (contingency).</b> The same quantity of dopamine delivered non-contingently "
 "(yoked) does not build desire &mdash; it is the pairing, not the chemical, that matters.",
 "<b>H4 (lability + directedness).</b> Desire extinguishes when reward stops, "
 "reinstates when it resumes, and assigns synaptic credit preferentially to the "
 "pathway that earned it.",
])

# =====================================================================
# 2 BACKGROUND
# =====================================================================
h1("2. Background and theory")
h2("2.1 Wanting is not liking")
p("Berridge and Robinson (1998) dissociated two components of reward that folk "
  "psychology fuses. <b>Liking</b> is the hedonic reaction to a reward at the moment "
  "of consumption; it is mediated by opioid/endocannabinoid &lsquo;hedonic "
  "hotspots&rsquo; and survives dopamine depletion. <b>Wanting</b> &mdash; incentive "
  "salience &mdash; is the motivational magnet that a reward-predicting cue acquires: "
  "it is what makes an animal <i>pursue</i>, and it is dopamine-dependent. A "
  "dopamine-depleted rat still shows normal &lsquo;liking&rsquo; facial reactions to "
  "sucrose but will not work for it. Desire, in this study, is wanting.")
h2("2.2 Dopamine as a teaching signal (reward-prediction error)")
p("Schultz, Dayan and Montague (1997) showed midbrain dopamine neurons encode a "
  "reward-<i>prediction error</i>: they burst to unexpected reward, fall silent to "
  "fully predicted reward, and dip below baseline when an expected reward is omitted. "
  "That dip is the teaching signal of extinction, and this study uses it directly to "
  "extinguish acquired pursuit.")
h2("2.3 The substrate in BioNeuron")
p("BioNeuron already contains the machinery incentive salience requires: a "
  "three-factor learning rule (Fr&eacute;maux &amp; Gerstner 2016; Izhikevich 2007). "
  "Recently co-active synapses are tagged by a slow <i>eligibility trace</i>; a later "
  "global dopamine signal converts the tag into a weight change:")
code("M = da_gain * (dopamine - 1) - cort_gain * cortisol\n"
     "dW = neuromod_lr * M * eligibility        (per eligible synapse)")
p("A cue-&gt;action projection that is repeatedly active-then-rewarded is therefore "
  "potentiated, so the cue progressively acquires the power to drive the action. "
  "That <i>is</i> incentive salience, mechanised: no new code, only the existing "
  "reward path exercised as a motivational assay.")

# =====================================================================
# 3 PROCEDURE
# =====================================================================
h1("3. Procedure")
h2("3.1 The incentive pathway")
p("Each network (N = %d) is given two initially identical, weak projections "
  "(w = 0.30): a <b>cue-&gt;action</b> pathway (20 &lsquo;cue&rsquo; neurons onto 30 "
  "&lsquo;action&rsquo; neurons) that will be driven and rewarded, and a parallel "
  "<b>control</b> pathway (ctrl-&gt;ctrl-out) onto a <i>separate, never-driven, "
  "never-rewarded</i> output group. The control pathway is the yardstick for "
  "reward specificity. Structural growth and pruning are frozen so only <i>weights</i> "
  "learn; the pathway neurons are set excitatory so the cue drive is unambiguous." % DATA['N'])
h2("3.2 A trial")
p("A <b>presentation</b> drives the cue group for 60 steps and then runs 40 steps "
  "unstimulated (dt = 0.1 ms); the total spikes emitted by the action group over that "
  "window is the network&rsquo;s <b>cue-evoked pursuit</b>. A <b>probe</b> is the mean "
  "pursuit over six presentations with dopamine at baseline &mdash; at baseline the "
  "consolidation signal M = 0, so probing measures without teaching. Reward is a "
  "phasic dopamine burst (reward-modulated three-factor LTP); reward omission is a "
  "phasic dopamine dip (negative prediction error -&gt; LTD).")
h2("3.3 Phases (run in order, per network)")
p("warm-up -&gt; <b>na&iuml;ve</b> probe -&gt; rewarded <b>training</b> (reward when "
  "cue-evoked pursuit reaches the operant criterion) -&gt; <b>trained</b> probe -&gt; "
  "<b>extinction</b> (cue presented, reward withdrawn, dopamine dip) -&gt; "
  "<b>extinguished</b> probe -&gt; <b>reinstatement</b> (reward re-paired with the "
  "cue) -&gt; <b>reinstated</b> probe.")
h2("3.4 Conditions (one intact case, two controls)")
tbl(["Condition", "What is changed", "Models"],
    [["<b>INTACT</b>", "dopamine-competent (da_gain = 1); reward contingent on the "
      "cue-&gt;action pathway", "a normal, motivated brain"],
     ["<b>DA-LESION</b>", "da_gain = 0: dopamine can no longer gate plasticity; "
      "spiking and reward delivery are otherwise identical", "dopamine depletion "
      "(wanting abolished, movement/hedonics spared)"],
     ["<b>DECOUPLED</b>", "same dopamine bursts, delivered non-contingently &mdash; "
      "the eligibility (credit) trace is cleared before each reward so it is unpaired "
      "with the action", "a yoked control (reinforcer present but not earned)"]],
    [1.05*inch, 3.35*inch, 1.8*inch])
cap("The two controls fail for different reasons &mdash; no usable dopamine signal "
    "(DA-LESION) versus dopamine present but non-contingent (DECOUPLED) &mdash; so an "
    "intact network that beats both is showing that desire needs the signal AND the pairing.")

story.append(PageBreak())

# =====================================================================
# 4 METHODOLOGY (markers + aggregation)
# =====================================================================
h1("4. Methodology: the four markers")
p("Each marker is a scalar, oriented so that higher = more desire-like, and each is "
  "grounded in an established property of incentive motivation.")
tbl(["Marker", "Definition", "Grounded in"],
    [["<b>Incentive salience</b>", "trained &ndash; na&iuml;ve pursuit (how much "
      "cue-evoked pursuit the reward built)", "Berridge &amp; Robinson 1998"],
     ["<b>Pursuit asymptote</b>", "trained pursuit level (the magnitude of acquired "
      "wanting)", "incentive-salience magnitude"],
     ["<b>Reinstatement</b>", "reinstated &ndash; extinguished pursuit (can the "
      "wanting be revived?)", "Bouton 2004; Schultz 1997"],
     ["<b>Directed credit</b>", "dW(cue-&gt;action) &ndash; dW(ctrl-&gt;"
      "ctrl-out): reward strengthens the earning pathway more than an unpaired one",
      "Izhikevich 2007; Schultz 1997"]],
    [1.25*inch, 3.55*inch, 1.4*inch])
p("<b>Aggregate desire index.</b> Exactly as in the awareness battery, the four "
  "markers are z-scored across the three conditions (removing their different units) "
  "and averaged into a single index per condition. A clean result is INTACT on top "
  "with both controls below zero. Reported values are mean &plusmn; s.d. over "
  f"{len(DATA['seeds'])} seeds ({', '.join(map(str, DATA['seeds']))}).")
h2("4.1 Key parameters")
code("enable_reward_learning = True;  neuromod_lr = 0.02;  tau_elig = 300 ms\n"
     "tau_dopamine = 200 ms;  w_max = 2.0;  noise_std = 1.8 (sqrt-dt)\n"
     "training = 60 trials;  extinction = 30;  reinstatement = 12;  probe = 6 presentations")

# =====================================================================
# 5 RESULTS
# =====================================================================
h1("5. Results")
p("Every hypothesis is supported. The intact network learns to want the cue; both "
  "controls do not; and the intact network dissociates on every individual marker and "
  "on the aggregate index.")

h2("5.1 Acquisition of incentive salience (H1, H2, H3)")
fig('fig1_acquisition.png')
cap("Figure 1. Cue-evoked pursuit across 60 rewarded training trials (mean over seeds). "
    "Only the INTACT network climbs &mdash; from the na&iuml;ve baseline to several-fold "
    "higher. DA-LESION (no dopamine gating) and DECOUPLED (non-contingent reward) stay flat: "
    "the same activity and the same dopamine, but no wanting is built.")
p("Averaged over seeds, INTACT pursuit rises from a na&iuml;ve %s to a trained "
  "%s spikes/trial, an incentive-salience gain of <b>%s</b>. The controls do not "
  "acquire (gain %s for DA-LESION, %s for DECOUPLED)."
  % (fmt(*ph('INTACT','naive'),1), fmt(*ph('INTACT','trained'),1),
     fmt(*mm('INTACT','salience_gain'),1), fmt(*mm('DA-LESION','salience_gain'),1),
     fmt(*mm('DECOUPLED','salience_gain'),1)))

h2("5.2 The desire life-cycle: extinction and reinstatement (H4)")
fig('fig2_timeline.png')
cap("Figure 2. The full life-cycle of an acquired desire (mean &plusmn; s.d.). INTACT "
    "acquires pursuit, loses it under extinction (reward withdrawn -&gt; dopamine dip "
    "-&gt; LTD), and recovers it on reinstatement when reward is re-paired with the cue. "
    "The controls have no pursuit to lose or revive.")
p("Reinstatement &mdash; the recovery of pursuit once the cue&ndash;reward pairing "
  "resumes &mdash; is <b>%s</b> spikes/trial for INTACT versus %s and %s for the "
  "controls: the intact network&rsquo;s wanting is <i>labile but revivable</i>, the "
  "signature of a motivational state rather than a fixed reflex."
  % (fmt(*mm('INTACT','reinstatement'),1), fmt(*mm('DA-LESION','reinstatement'),1),
     fmt(*mm('DECOUPLED','reinstatement'),1)))

h2("5.3 Markers and the aggregate desire index")
fig('fig3_markers.png', width=6.0*inch)
cap("Figure 3. The four desire markers by condition (mean &plusmn; s.d.). INTACT leads "
    "on all four. Note the scales differ &mdash; the aggregate index (Figure 4) puts them "
    "on a common footing.")
# results table straight from the data
rows = []
for m in MARKS:
    rows.append([MLAB[m]] + [fmt(*mm(c, m), 2 if m == 'directed_credit' else 1) for c in CONDS])
rows.append(["<b>DESIRE INDEX (z-avg)</b>"] + ["<b>%s</b>" % fmt(*di(c), 2) for c in CONDS])
tbl(["marker (mean &plusmn; s.d.)"] + CONDS, rows,
    [2.35*inch, 1.4*inch, 1.4*inch, 1.4*inch],
    align={1:"CENTER",2:"CENTER",3:"CENTER"})
fig('fig4_index.png', width=4.4*inch)
cap("Figure 4. Aggregate desire index. INTACT separates cleanly and positively; both "
    "controls fall below zero. This is the headline dissociation.")

h2("5.4 Directed credit and the global-dopamine limit")
fig('fig5_credit.png', width=5.0*inch)
cap("Figure 5. Potentiation of the reward-paired pathway (cue-&gt;action) versus the "
    "unpaired control pathway. INTACT potentiates the paired pathway most, but the "
    "control pathway also grows &mdash; see the note below.")
p("The reward-paired cue-&gt;action pathway is potentiated by "
  "dW = %s in INTACT, against %s for DA-LESION and %s for DECOUPLED. Credit is "
  "therefore <i>directed</i>: reward strengthens the pathway that earned it. But an "
  "honest reading of Figure 5 shows the control pathway <i>also</i> grows in INTACT "
  "(dW = %s). This is not noise &mdash; it is the credit-assignment problem: "
  "dopamine is a <b>global</b> broadcast, so every synapse that happens to be eligible "
  "near a reward receives some potentiation. The eligibility trace confers "
  "<i>partial</i>, not perfect, specificity. We report this rather than hide it; it is "
  "a real property of biological three-factor learning and a target for future "
  "credit-sharpening (e.g. attention-gated or compartmentalised eligibility)."
  % (fmt(*AGG['INTACT']['dW_cue'],2), fmt(*AGG['DA-LESION']['dW_cue'],2),
     fmt(*AGG['DECOUPLED']['dW_cue'],2), fmt(*AGG['INTACT']['dW_ctrl'],2)))

# =====================================================================
# 6 DISCUSSION
# =====================================================================
h1("6. Discussion")
p("The pattern is exactly the one incentive-salience theory predicts. Desire in this "
  "network is not a built-in parameter; it is <i>constructed</i> by experience through "
  "dopamine-gated plasticity, and it decomposes along the same fault lines as animal "
  "wanting. Two dissociations carry the argument.")
bullets([
 "<b>Dopamine-dependence (INTACT vs DA-LESION).</b> With da_gain = 0 the network still "
 "spikes, still receives the cue, still &lsquo;acts&rsquo; &mdash; and builds no "
 "pursuit whatsoever. Movement without wanting is precisely the dopamine-depletion "
 "phenotype (Berridge 2007). Desire here lives in the dopamine gate, not in the activity.",
 "<b>Contingency (INTACT vs DECOUPLED).</b> DECOUPLED receives the identical quantity "
 "of dopamine, merely unpaired from the action. It does not learn to want. Desire is "
 "built by the <i>earned</i> pairing of action and reward, not by the presence of the "
 "reward chemical &mdash; the computational content of &lsquo;the reward must be "
 "contingent on behaviour&rsquo;.",
 "<b>Lability (extinction &amp; reinstatement).</b> The acquired wanting is not a "
 "frozen weight; it decays when reward stops and returns when it resumes, the hallmark "
 "of a motivational <i>state</i>.",
])
p("The credit-leakage finding (5.4) is, if anything, a mark of biological realism: a "
  "single global neuromodulator cannot perfectly address individual synapses, and real "
  "brains face and partly solve the same problem. It bounds the strength of the "
  "&lsquo;directed&rsquo; claim without undermining it.")

# =====================================================================
# 7 LIMITATIONS / SCOPE
# =====================================================================
h1("7. Limitations and scope")
bullets([
 "<b>This does not measure felt desire.</b> Every marker is a functional signature. "
 "As with the awareness battery, no external measurement can establish whether "
 "anything is subjectively wanted; a high desire index means desire-like <i>dynamics</i>, "
 "read only relative to the controls.",
 "<b>Wanting only, not liking.</b> The battery targets incentive salience. The "
 "consummatory/hedonic component (&lsquo;liking&rsquo;, opioidergic) is not modelled here.",
 "<b>Small and abstract.</b> N = %d, a single cue and a single action, on CPU. The "
 "cue&ndash;action mapping is a stand-in for a goal, not an ecological behaviour." % DATA['N'],
 "<b>Partial credit specificity.</b> Global dopamine potentiates unpaired eligible "
 "synapses too (5.4); the &lsquo;directed&rsquo; marker is a difference, not an absolute.",
 "<b>Hand-set operant criterion.</b> The reward threshold (half the na&iuml;ve rate) "
 "was chosen so acquisition reliably bootstraps; it is held identical across conditions "
 "so it cannot manufacture the dissociation, but it is a designed, not discovered, value.",
])

# =====================================================================
# 8 CONCLUSION
# =====================================================================
h1("8. Conclusion")
p("BioNeuron exhibits the operational markers of desire. An intact, dopamine-competent "
  "network learns to pursue a reward-predicting cue, sustains and revives that pursuit, "
  "and directs synaptic credit toward the pathway that earned reward &mdash; while a "
  "dopamine-lesioned network and a reward-decoupled network, matched on activity and on "
  "dopamine quantity respectively, do neither. Desire, so operationalised, is an "
  "emergent and dissociable consequence of dopamine-gated three-factor plasticity, "
  f"separating on the aggregate index at {di('INTACT')[0]:+.2f} against "
  f"{di('DA-LESION')[0]:+.2f} and {di('DECOUPLED')[0]:+.2f}. Whether the network "
  "<i>feels</i> the wanting it so clearly enacts is a question this &mdash; or any "
  "&mdash; battery is constitutively unable to answer.")

# =====================================================================
# 9 REFERENCES
# =====================================================================
h1("9. References")
refs = [
 "Berridge, K. C., and Robinson, T. E. (1998). What is the role of dopamine in reward: "
 "hedonic impact, reward learning, or incentive salience? Brain Research Reviews 28(3), 309-369.",
 "Berridge, K. C. (2007). The debate over dopamine's role in reward: the case for "
 "incentive salience. Psychopharmacology 191(3), 391-431.",
 "Bouton, M. E. (2004). Context and behavioral processes in extinction. Learning &amp; "
 "Memory 11(5), 485-494.",
 "Frémaux, N., and Gerstner, W. (2016). Neuromodulated spike-timing-dependent "
 "plasticity, and theory of three-factor learning rules. Frontiers in Neural Circuits 9, 85.",
 "Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP "
 "and dopamine signaling. Cerebral Cortex 17(10), 2443-2452.",
 "Legenstein, R., Pecevski, D., and Maass, W. (2008). A learning theory for "
 "reward-modulated spike-timing-dependent plasticity with application to biofeedback. "
 "PLoS Computational Biology 4(10), e1000180.",
 "Schultz, W., Dayan, P., and Montague, P. R. (1997). A neural substrate of prediction "
 "and reward. Science 275(5306), 1593-1599.",
 "Tononi, G., and Koch, C. (2015). Consciousness: here, there and everywhere? "
 "Philosophical Transactions of the Royal Society B 370(1668), 20140167. "
 "(on the limits of behavioural/dynamical markers of inner states.)",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph("%d. %s" % (i, r), ref_s))

doc = SimpleDocTemplate(OUT, pagesize=letter,
                        leftMargin=72, rightMargin=72, topMargin=64, bottomMargin=64,
                        title="The Desire Battery - operational markers of incentive motivation",
                        author="Jugal Kishore Nagulapalli")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("WROTE", OUT)
