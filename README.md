# Funkcijos optimizavimas (GA ir MA)

Ši programa ieško funkcijos „Trys kalniukai“ aukščiausio taško (maksimumo) naudojant evoliucinius algoritmus.

## Reikalavimai
Prieš paleisdami kodą, įsidiekite reikiamas bibliotekas:
```bash
pip install numpy matplotlib scipy
```
## Algoritmų paleidimai
```bash
    best_bin_ma, best_fit_ma, history_ma, iterations_ma = memetic_genetic_algorithm_D(
        initial_parents=initial_pop,
        fitness_function=fitness_D,
        mutation_rate=0.05,
        population_size=POP_SIZE, 
        D=current_D,
        generations=MAX_GENS)

    best_bin_ma, best_fit_ma, history_ma, iterations_ma = genetic_algorithm_D(
        initial_parents=initial_pop,
        fitness_function=fitness_D,
        mutation_rate=0.05,
        population_size=POP_SIZE,  
        D=current_D,
        generations=MAX_GENS)
```

