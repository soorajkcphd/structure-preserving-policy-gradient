.PHONY: all selftest theory mutation figures quick clean
all: selftest theory mutation figures
selftest: ; cd .. && python -m sppg_defense.selftest
theory:   ; cd .. && python -m sppg_defense.run_theory
mutation: ; cd .. && python -m sppg_defense.mutation_test
quick:    ; cd .. && python -m sppg_defense.run_theory --quick
figures:  ; cd .. && python -m sppg_defense.figures.make_figures
clean:    ; rm -f results/*.csv results/*.json figures/out/*
