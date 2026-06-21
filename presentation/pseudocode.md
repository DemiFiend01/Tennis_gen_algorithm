# Genetic Algorithm Pseudocode
## Atari Tennis — NeuroEvolution

---

### Parameters
```
no_inv          ← number of individuals in population
no_gen          ← number of generations
parents_frac    ← fraction of population selected for mating
max_steps       ← maximum simulation steps per individual
max_time        ← maximum wall-clock time per evaluation (seconds)
```

---

### 1. Initialise Population

```
FUNCTION init_population(no_inv):
    population ← []

    FOR i = 1 TO no_inv DO
        Build NN with layers: n_x=11 → h1=32 → h2=16 → n_y=10
        Xavier-init weight matrices W1, W2, W3
        population.append(flatten([W1, W2, W3]))
    END FOR

    RETURN population
```

---

### 2. Main Loop

```
population ← init_population(no_inv)
generation ← 0

WHILE generation < no_gen DO

    generation ← generation + 1

    ── 2a. Evaluate Population ──────────────────────────────────

    FOR each individual in population DO

        Create OCAtari "Tennis-v4" env (RAM mode, no render)
        Rebuild Individual from weight vector
        service_now ← True

        WHILE not terminated DO

            state ← extract 11-value game state from RAM

            IF service_now THEN
                action ← 1  (force serve)
                IF ball started moving THEN
                    service_now ← False
                END IF

            ELSE
                h1     ← ReLU(state @ W1)
                h2     ← ReLU(h1    @ W2)
                logits ← h2 @ W3
                action ← argmax(softmax(logits)) + 1
            END IF

            Execute action in env → observe RAM delta

            Track metrics:
                avg_player_ball_dist  ← distance(player, ball)
                                        when ball moves toward player
                player_score, enemy_score
                step_counter ← step_counter + 1

            IF score changed OR deadlock detected THEN
                service_now ← True
            END IF

            IF elapsed_time ≥ max_time OR step_counter ≥ max_steps THEN
                BREAK  (early termination)
            END IF

        END WHILE

        Compute fitness f:
            f ← 1 − avg_player_ball_dist
            IF generation > 10 THEN  f ← f + (step_counter / max_steps)
            IF generation > 20 THEN  f ← f + (1 − avg_missed_distance)
            IF generation > 30 THEN  f ← f + (player_score / total_points)

        Store f as fitness of this individual

    END FOR

    ── 2b. On Generation Callback ───────────────────────────────

    Print best fitness of current generation

    IF generation MOD 5 = 0 THEN
        Save best weight vector to /tmp/best_fits/gen-{g}-fit-{f}.npy
    END IF

    ── 2c. Selection → Crossover → Mutation ─────────────────────

    Rank population by fitness
    elite ← top-1 individual  (preserved unchanged)

    parents_mating ← floor(no_inv × parents_frac)
    parents ← SSS_select(population, parents_mating)

    new_population ← [elite]

    WHILE size(new_population) < no_inv DO

        parent_A, parent_B ← pick two parents from parents

        // Uniform crossover
        FOR each gene position DO
            child_gene ← gene from parent_A or parent_B  (50/50)
        END FOR

        // Random mutation
        For mutation_percent_genes % of genes:
            gene ← gene + uniform(−0.15, 0.15)

        new_population.append(child)

    END WHILE

    population ← new_population

END WHILE
```

---

### 3. Output

```
RETURN individual with highest recorded fitness
       (best weight vector across all generations)
```