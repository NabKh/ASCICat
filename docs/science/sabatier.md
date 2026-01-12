# Sabatier Principle

The foundation of activity scoring in ASCICat.

## Historical Background

Paul Sabatier (Nobel Prize 1912) proposed:

> *"An ideal catalyst binds reaction intermediates with intermediate strength."*

Too weak binding → reactants don't stick
Too strong binding → products don't desorb

## The Volcano Plot

Plotting catalytic activity vs. binding energy produces a "volcano":

```
Activity
   ↑
   |           ★ Optimum
   |          /|\
   |         / | \
   |        /  |  \
   |    Pt /   |   \ Au
   |      /    |    \
   | Ni  /     |     \ Ag
   |____/______|______\____→ Binding Energy
    strong    optimal    weak
```

## Theoretical Basis

### Brønsted-Evans-Polanyi (BEP) Relations

Activation energy correlates with reaction energy:

$$E_a = E_a^0 + \alpha \Delta E$$

- Stronger binding → Lower adsorption barrier
- Stronger binding → Higher desorption barrier

### Scaling Relations

Different intermediates scale together:

$$\Delta E_{*OH} = a \cdot \Delta E_{*O} + b$$

This enables single-descriptor screening.

## Application to HER

For Hydrogen Evolution Reaction:

$$\Delta E_{opt} = -0.27 \text{ eV}$$

**Derivation:**

1. Thermoneutral point: $\Delta G_{H*} \approx 0$
2. ZPE and entropy corrections: $\Delta E_{H*} \approx -0.27$ eV
3. Validated by Pt(111) experiments

## Application to CO2RR

| Pathway | Intermediate | $\Delta E_{opt}$ |
|:--------|:-------------|:-----------------|
| CO | *CO | -0.67 eV |
| CHO | *CHO | -0.48 eV |
| COCOH | *COOH | -0.32 eV |

## Linear Scoring Implementation

$$S_a = \max\left(0, 1 - \frac{|\Delta E - \Delta E_{opt}|}{\sigma_a}\right)$$

**Parameters:**

- $\Delta E_{opt}$ = Volcano peak position
- $\sigma_a$ = Width parameter (typically 0.15 eV)

## Why 0.15 eV?

The width parameter accounts for:

1. **DFT uncertainty** (~0.1 eV)
2. **Functional variation** (~0.05-0.1 eV)
3. **Surface coverage effects**

## References

- Sabatier, P. *Ber. Dtsch. Chem. Ges.* **44**, 1984 (1911)
- Nørskov, J. K. et al. *J. Catal.* **209**, 275 (2002)
- Greeley, J. et al. *Nat. Mater.* **5**, 909 (2006)
