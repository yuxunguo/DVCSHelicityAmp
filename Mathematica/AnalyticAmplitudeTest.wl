(* ::Package:: *)

(* Print one fully contracted analytic Bethe-Heitler helicity amplitude.

   All kinematic quantities remain symbolic. By default the independent
   variables are

     {s, thetaIn, phiIn, eGamma, phiGamma, m, F1t, F2t}.

   Edit helicityInputs below to select

     {hIn, hOut, sIn, sOut, lambda}.

   Run from the repository root:

     /Applications/Wolfram.app/Contents/MacOS/WolframKernel \
       -script Mathematica/AnalyticAmplitudeTest.wl

   Pass --summary to construct and validate the expression without printing
   the complete (very large) formula:

     /Applications/Wolfram.app/Contents/MacOS/WolframKernel \
       -script Mathematica/AnalyticAmplitudeTest.wl --summary
*)

SetDirectory[NotebookDirectory[]]
<<"BHHelicityAmp.wl"

(* Doubled-helicity convention: every entry must be -1 or +1. *)
helicityInputs = {-1, -1, -1, 1, 1};
{hIn, hOut, sIn, sOut, lambda} = helicityInputs;

(* False preserves the much more readable contracted sum. Set True to force
   the entire result over one common denominator; this can take substantially
   longer for unrestricted general-angle kinematics. *)
combineDenominators = False;

If[!AllTrue[helicityInputs, MemberQ[{-1, 1}, #] &],
  Print["Invalid helicity input. Use only -1 or +1."];
  Exit[1];
];

(* User-frame kinematics with pIn and the physical pOut root expressed
   analytically in terms of the independent variables. *)
kinematics = UserKinematics[
  s, thetaIn, phiIn, eGamma, phiGamma, m
];

amplitude = BHAmplitude[
  kinematics,
  hIn, hOut, sIn, sOut, lambda,
  m, F1t, F2t
];

(* BHAmplitude has already performed every Dirac-matrix multiplication,
   spinor contraction, photon-polarization contraction, and Lorentz sum. *)
analyticAmplitude = If[combineDenominators, Together[amplitude], amplitude];

commandLineStrings = ToString /@ Join[$ScriptCommandLine, $CommandLine];
summaryOnly = AnyTrue[commandLineStrings, StringContainsQ[#, "--summary"] &];

Print["Helicities {hIn,hOut,sIn,sOut,lambda} = ", helicityInputs];
Print["Independent variables = ",
  {s, thetaIn, phiIn, eGamma, phiGamma, m, F1t, F2t}];
Print["Expression head = ", Head[analyticAmplitude]];
Print["Expression leaf count = ", LeafCount[analyticAmplitude]];
Print["Common denominator requested = ", combineDenominators];
Print["Uncontracted matrix products remaining = ",
  Count[analyticAmplitude, _Dot, Infinity]];

If[!summaryOnly,
  Print["\nM[hIn,hOut,sIn,sOut,lambda] ="];
  Print[InputForm[analyticAmplitude]];
];



