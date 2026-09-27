\# Experiments



This directory contains intermediate experiments conducted during development of

the Business Entity Resolution solution.



\## Blocking



The `blocking/` directory contains experiments for evaluating different

candidate-generation and blocking strategies.



These experiments were used to study:



\- Candidate recall

\- Candidate precision

\- Candidate volume

\- Blocking-key combinations

\- Candidate caps

\- Missed true matches

\- Address and name based blocking signals



The blocking experiments evolved through multiple versions before the final

candidate-generation strategy was selected.



\## Pipeline



The `pipeline/` directory contains intermediate scripts used for:



\- Candidate generation

\- Submission generation

\- Candidate analysis

\- Ground-truth inspection

\- Intermediate blocking evaluation



\## Model



The `model/` directory contains intermediate model/inference scripts that were

used during development but are not required by the final production pipeline.



\## Final Implementation



The final reproducible implementation is maintained under:



`../src/`



The files under `src/` represent the final solution pipeline used to generate

the submitted results.

