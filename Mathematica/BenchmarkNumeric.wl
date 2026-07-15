(* ::Package:: *)

(* Evaluate numerical kinematics and helicity-amplitude benchmarks with an
   explicit physical electron mass. Change Ml to benchmark another lepton. *)
SetDirectory[DirectoryName[ExpandFileName[$InputFileName]]];
<<"BHHelicityAmp.wl";

helicities = {-1, 1};

(* Independent user-frame inputs copied from Output/BHHelicityAmp.log. *)
benchmarkInputs = {
  <|"case" -> "K1", "s" -> 10.25844, "thetaIn" -> 1.10,
    "phiIn" -> 0.20, "EGamma" -> 0.45, "phiGamma" -> 2.40,
    "Mp" -> 0.938, "Ml" -> 0.00051099895|>,
  <|"case" -> "K2", "s" -> 14.01544, "thetaIn" -> 1.45,
    "phiIn" -> 0.70, "EGamma" -> 0.70, "phiGamma" -> 3.10,
    "Mp" -> 0.938, "Ml" -> 0.00051099895|>,
  <|"case" -> "K3", "s" -> 19.64894, "thetaIn" -> 1.90,
    "phiIn" -> 1.10, "EGamma" -> 0.95, "phiGamma" -> 3.70,
    "Mp" -> 0.938, "Ml" -> 0.00051099895|>
};

(* Form-factor rows ff=1,...,6 from the log. *)
formFactors = {
  {0.5, 0.0}, {0.8, 0.0}, {1.0, 0.2},
  {1.0, -0.2}, {0.7, 0.5}, {0.0, 1.0}
};

makeCase[input_] := Module[{kin, Mp, Ml, amplitudeF1, amplitudeF2},
  Mp = input["Mp"]; Ml = input["Ml"];
  kin = UserKinematics[
    input["s"], input["thetaIn"], input["phiIn"],
    input["EGamma"], input["phiGamma"], Mp, Ml
  ];

  (* The BH amplitude is linear in F1 and F2. These two basis tables allow
     every form-factor row to be reconstructed without repeating the Dirac
     contractions six times. *)
  amplitudeF1 = BHAmplitudeTable[kin, Mp, Ml, 1.0, 0.0];
  amplitudeF2 = BHAmplitudeTable[kin, Mp, Ml, 0.0, 1.0];

  <|"input" -> input, "kin" -> kin,
    "amplitudeF1" -> amplitudeF1, "amplitudeF2" -> amplitudeF2|>
];

cases = Association@Table[
  input["case"] -> makeCase[input],
  {input, benchmarkInputs}
];

amplitudeFor[data_, f1_, f2_] :=
  f1 data["amplitudeF1"] + f2 data["amplitudeF2"];

unpolarizedM2[amplitudes_] := Total[Abs[Flatten[amplitudes]]^2]/4;

(* Row four is (hIn,sIn)=(+1,+1) in the {-1,+1} basis. *)
fixedPositiveM2[amplitudes_] := Total[Abs[amplitudes[[4]]]^2];

Print["\nIndependent user-frame inputs"];
Print[TableForm[
  ({#["case"], #["s"], #["thetaIn"], #["phiIn"],
      #["EGamma"], #["phiGamma"], #["Mp"], #["Ml"]} &) /@ benchmarkInputs,
  TableHeadings -> {None,
    {"case", "s", "thetaIn", "phiIn", "EGamma", "phiGamma", "Mp", "Ml"}}
]];

Print["\nDerived kinematics and invariant diagnostics"];
Print[TableForm[
  Table[
    With[{kin = cases[label]["kin"]},
      {label, Sqrt[kin["s"]], kin["pIn"], kin["pOut"],
        kin["Q2"], kin["xB"], kin["t"], kin["W2"], kin["y"]}],
    {label, Keys[cases]}
  ] // N,
  TableHeadings -> {None,
    {"case", "sqrt(s)", "pIn", "pOut", "Q2", "xB", "t", "W2", "y"}}
]];

Print["\nFour-momenta [E,px,py,pz]"];
Do[
  Print[label, "  ",
    AssociationMap[N[cases[label]["kin"][#]] &,
      {"k", "p", "kp", "pp", "qout", "q"}]
  ],
  {label, Keys[cases]}
];

Print["\nKinematic checks"];
Do[
  Print[label, "  ", N[KinematicChecks[
    cases[label]["kin"], cases[label]["input"]["Mp"],
    cases[label]["input"]["Ml"]]]],
  {label, Keys[cases]}
];

Print["\nUnpolarized (1/4) sum |M|^2"];
unpolarizedRows = Flatten[Table[
  With[{f1 = formFactors[[ff, 1]], f2 = formFactors[[ff, 2]]},
    With[{amplitudes = amplitudeFor[cases[label], f1, f2]},
      {label, ff, f1, f2, unpolarizedM2[amplitudes]}]],
  {label, Keys[cases]},
  {ff, Length[formFactors]}
], 1];
Print[TableForm[N[unpolarizedRows],
  TableHeadings -> {None, {"kin", "ff", "F1", "F2", "unpol |M|^2"}}
]];

Print["\nFixed (hIn,sIn)=(+1,+1) sum over final helicities"];
polarizedRows = Flatten[Table[
  With[{f1 = formFactors[[ff, 1]], f2 = formFactors[[ff, 2]]},
    With[{amplitudes = amplitudeFor[cases[label], f1, f2]},
      {label, ff, f1, f2, fixedPositiveM2[amplitudes]}]],
  {label, Keys[cases]},
  {ff, Length[formFactors]}
], 1];
Print[TableForm[N[polarizedRows],
  TableHeadings -> {None,
    {"kin", "ff", "F1", "F2", "fixed h,S |M|^2"}}
]];

(* Match the final amplitude table in the log: K1 at F1=1,F2=0. *)
referenceAmplitudes = amplitudeFor[cases["K1"], 1.0, 0.0];
helicityPosition[h_] := If[h == -1, 1, 2];
incomingIndex[hIn_, sIn_] :=
  1 + 2 (helicityPosition[hIn] - 1) + helicityPosition[sIn] - 1;
outgoingIndex[hOut_, sOut_, lambda_] :=
  1 + 4 (helicityPosition[hOut] - 1) +
    2 (helicityPosition[sOut] - 1) + helicityPosition[lambda] - 1;

fixedHelicityRows = Flatten[Table[
  With[{amplitude = referenceAmplitudes[[
      incomingIndex[hIn, sIn], outgoingIndex[hOut, sOut, lambda]]]},
    {hIn, hOut, sIn, sOut, lambda,
      Re[amplitude], Im[amplitude], Abs[amplitude]^2}],
  {hIn, helicities}, {hOut, helicities}, {sIn, helicities},
  {sOut, helicities}, {lambda, helicities}
], 4];

Print["\nFixed-helicity amplitudes for K1 at F1=1,F2=0"];
Print[TableForm[N[fixedHelicityRows],
  TableHeadings -> {None,
    {"hIn", "hOut", "sIn", "sOut", "lambda", "Re M", "Im M", "|M|^2"}}
]];

(* Density matrices and concurrence based on the same K1 reference table. *)
rhoInUnpolarized = InitialSpinDensity["Unpolarized", "Unpolarized"];
rhoOutUnpolarized = OutgoingDensityMatrix[
  referenceAmplitudes, rhoInUnpolarized];

rhoInLL = InitialSpinDensity["L", "L"];
rhoOutLL = OutgoingDensityMatrix[referenceAmplitudes, rhoInLL];

Print["\nDensity matrix and concurrence for K1 at F1=1,F2=0"];
Print["Tr[rhoOut unpolarized] = ", Chop[Tr[rhoOutUnpolarized]]];
Print["unpolarized observables = ",
  EntanglementObservables[rhoOutUnpolarized]];
Print["Tr[rhoOut LL] = ", Chop[Tr[rhoOutLL]]];
Print["LL observables = ", EntanglementObservables[rhoOutLL]];

