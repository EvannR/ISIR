BICA 2026 Paper Submission - Rabeau & Iste

This is an adapted version of the paper originally submitted to EUMAS 2026.

KEY ADAPTATIONS FOR BICA 2026:
==============================

1. DOCUMENT CLASS & FORMATTING
   - Uses LLNCS (Springer) document class [runningheads]
   - Compatible with BICA 2026 formatting requirements

2. AUTHOR INFORMATION
   - Added ORCID IDs for all authors (marked as 0000-0000-0000-0000 as placeholders)
   - Update these with actual ORCID numbers before submission
   - Format: \orcidID{XXXX-XXXX-XXXX-XXXX}

3. BIBLIOGRAPHY STYLE
   - Uses thebibliography environment (inline bibliography)
   - All references are embedded in the main.tex file
   - The ref.bib file is included for reference but not used for compilation

4. ACKNOWLEDGMENTS & DECLARATIONS
   - Acknowledgments are now in the \credits section with \ackname
   - Competing interests declaration is included with \discintname
   - This follows BICA 2026 submission guidelines

5. PACKAGES & STRUCTURE
   - Maintains amsmath package for mathematical notation
   - Float formatting parameters optimized for proceedings

FILE STRUCTURE:
===============
- main.tex              : Main document with all structure
- introduction.tex      : Introduction section (includes literature review)
- Method.tex            : Methods section with study design
- Results.tex           : Results section with figures
- Discussion.tex        : Discussion with appendices (simulation parameters and hyperparameters)
- llncs.cls             : LLNCS document class (Springer)
- splncs04.bst          : BibTeX style file
- fig1.eps              : Sample figure in EPS format
- figures/results/      : All result figures (PDF and PNG)
- ref.bib               : Bibliography database (for reference)

TO COMPILE:
===========
Option 1 (with pdflatex):
  pdflatex main.tex
  pdflatex main.tex  (run twice for correct references)

Option 2 (with BibTeX):
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex

NOTES:
======
- Ensure all figure paths are correct relative to the main.tex location
- The current main.tex uses inline bibliography (thebibliography), so bibtex is not required
- All authors should update ORCID IDs before final submission
- Check with BICA 2026 organizers for any additional formatting requirements
- The submission should be saved as PDF and uploaded to the conference system

This paper investigates the computational role of empathy in multi-agent reinforcement learning
for common-pool resource dilemmas, examining how empathy influences sustainability and fairness.
