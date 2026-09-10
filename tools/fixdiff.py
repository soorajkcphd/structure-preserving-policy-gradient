"""Repair latexdiff output: neutralise unbalanced \DIFdel runs, recolour, kill dead pointers."""
import re, sys

def balanced(t, start):
    d=1; j=start
    while j<len(t) and d>0:
        d += 1 if t[j]=='{' else (-1 if t[j]=='}' else 0); j+=1
    return d==0, j

def repair(t):
    """Replace every unbalanced \DIFdel{... run with a clean deletion marker."""
    n=0
    while True:
        hit=None
        for m in re.finditer(r'\\DIFdel\{', t):
            ok,_=balanced(t, m.end())
            if not ok: hit=m; break
        if hit is None: break
        # end the run at the next structural boundary
        ends=[t.find(k, hit.start()+1) for k in ('%DIFAUXCMD','\\DIFaddbegin','\\DIFdelend')]
        ends=[e for e in ends if e>0]
        e=min(ends) if ends else len(t)
        t = t[:hit.start()] + "\\DIFdel{[text deleted in revision]}\n" + t[e:]
        n+=1
        if n>20: raise SystemExit("too many unbalanced runs; inspect manually")
    return t, n

if len(sys.argv) != 3:
    sys.exit(__doc__.strip() + "\n\nusage: fixdiff.py <raw-latexdiff.tex> <output.tex>")

t = open(sys.argv[1]).read()
t,n = repair(t)
t=t.replace("\\providecommand{\\DIFaddtex}[1]{{\\protect\\color{blue} \\sf #1}}","\\providecommand{\\DIFaddtex}[1]{{\\protect\\color{red}#1}}")
t=t.replace("\\providecommand{\\DIFdeltex}[1]{{\\protect\\color{red} \\scriptsize #1}}","\\providecommand{\\DIFdeltex}[1]{{\\protect\\color{gray}\\scriptsize #1}}")
t=t.replace("\\newcommand{\\DIFaddincludegraphics}[2][]{{\\color{blue}","\\newcommand{\\DIFaddincludegraphics}[2][]{{\\color{red}")
t=t.replace("moredelim=[il][\\color{red}\\scriptsize]{\\%DIF\\ <\\ }","moredelim=[il][\\color{gray}\\scriptsize]{\\%DIF\\ <\\ }")
for r in ['eq:rel_residual','eq:structure_discovery','fig:structure_discovery','sec:lora_comparison',
          'tab:abl_combined','tab:lora_comparison','tab:perseed','tab:structure','tab:synthetic_ablation','tab:task2']:
    for c in ('ref','eqref'): t=t.replace('\\'+c+'{'+r+'}','\\mbox{--}')
for c in ['bottou2018optimization','davis2019stochastic','kc2026dichotomy','lojasiewicz1963','mei2020softmax']:
    t=t.replace('\\cite{'+c+'}','\\mbox{[--]}')
for a in ["\\cite[\\S4.3]{bottou2018optimization}","\\cite[Thms.~6.2--6.3]{kc2026dichotomy}","\\cite[Lem.~2]{mei2020softmax}"]:
    t=t.replace(a,"\\mbox{[--]}")
t=t.replace("\\cite{lojasiewicz1963,attouch2013convergence}","\\cite{attouch2013convergence}")
open(sys.argv[2],'w').write(t)
print(f"repaired {n} unbalanced deletion run(s)")
