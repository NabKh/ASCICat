# ASCICat Data File Guidelines

## Required Columns

All ASCICat input files must contain these three essential columns:

1. **DFT_ads_E** (float): Adsorption energy in eV
   - HER: H adsorption energy (optimal: -0.27 eV)
   - CO2RR-CO: CO adsorption energy (optimal: -0.67 eV)
   - CO2RR-CHO: CHO intermediate energy (optimal: -0.48 eV)
   - CO2RR-COCOH: COOH intermediate energy (optimal: -0.32 eV)

2. **surface_energy** (float): Surface energy in J/m²
   - Must be positive
   - Typical range: 0.5 - 5.0 J/m²
   - Lower is more stable

3. **Cost** (float): Material cost in $/kg
   - Must be positive
   - Typical range: $0.5 - $200,000/kg
   - Based on elemental composition

## Optional Columns

These columns enhance analysis but are not required:

- **symbol** (string): Catalyst identifier (highly recommended)
- **Ametal** (string): Primary metal element symbol
- **Bmetal** (string): Secondary metal element (for alloys)
- **Cmetal** (string): Tertiary metal element (for complex alloys)
- **slab_millers** (string): Miller indices of surface (e.g., "111", "100")
- **AandB** (string): Full surface description

## File Format

- **Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8
- **Header**: Required (first row with column names)
- **Missing values**: Not allowed in required columns

## Data Quality

ASCICat validates data automatically:

✓ **Checks performed:**
- Missing values in required columns
- Negative surface energies (invalid)
- Zero or negative costs (invalid)
- Extreme outliers (> 5 eV from optimal, > 15 J/m² surface energy)
- Duplicate entries

⚠️ **Common issues:**
- Mixing units (ensure eV for energy, J/m² for surface energy, $/kg for cost)
- Using adsorption energy instead of binding energy (sign convention)
- Missing or incorrect optimal energy reference

## Templates

See TEMPLATE_*.csv files for correctly formatted examples:
- `TEMPLATE_HER.csv` - Hydrogen Evolution Reaction
- `TEMPLATE_CO2RR_CO.csv` - CO2 to CO pathway
- `TEMPLATE_CO2RR_CHO.csv` - CO2 to methanol pathway
- `TEMPLATE_CO2RR_COCOH.csv` - CO2 to formic acid pathway

## Data Sources

Recommended sources for DFT data:
- Materials Project (materialsproject.org)
- Catalysis-Hub (catalysis-hub.org)
- C2DB (c2db.fysik.dtu.dk)
- AFLOW (aflowlib.org)

## Example
```csv
symbol,DFT_ads_E,surface_energy,Cost
Pt,-0.27,2.489,30000.0
Ni,-0.20,2.380,18.0
Cu,-0.35,1.825,8.5
```

## Questions?

Contact: n.khossossi@differ.nl
