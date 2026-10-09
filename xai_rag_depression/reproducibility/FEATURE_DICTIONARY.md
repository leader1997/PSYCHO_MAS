# Feature dictionary — HMS 2024–2025 pipeline (25 features)

| Feature | HMS construct | Original coding (typical) | Pipeline transform | Higher raw value means | Risk direction after scaling | Interpretation notes |
|---------|---------------|---------------------------|--------------------|------------------------|------------------------------|----------------------|
| `age` | Student age | Years | Median impute + z-score | Older | Context-dependent | Demographic covariate |
| `gender_male` | Gender | 1=male, 0=not | z-score | Male indicator | — | Mutually exclusive with gender_female in survey |
| `gender_female` | Gender | 1=female, 0=not | z-score | Female indicator | — | Demographic |
| `international` | International student | 1=international | z-score | International status | — | Demographic |
| `fincur` | Current finances | Higher = **better** financial situation (less strain) | z-score | **Less** financial stress | **Low** fincur → higher risk in SHAP | Protective when high; "low fincur increases risk" = worse finances |
| `finpast` | Past financial stress | Higher = more past strain (verify codebook) | z-score | More past stress | Higher → risk | Financial history |
| `food_worry` | Food insecurity worry | Higher = more worry | z-score | More food worry | Higher → risk | Basic needs stress |
| `sleep_wknight` | Weeknight sleep hours | Hours | z-score | More sleep | Protective if higher | Sleep quantity |
| `sleep_wkend` | Weekend sleep hours | Hours | z-score | More sleep | Protective if higher | Sleep quantity |
| `exerc` | Exercise frequency | Higher = more exercise | z-score | More exercise | Protective if higher | Lifestyle |
| `belong1` | Campus belonging (item 1) | Higher = more belonging | z-score | More belonging | Protective if higher | Social |
| `belong2` | Campus belonging (item 2) | Higher = more belonging | z-score | More belonging | Protective if higher | Social |
| `lonely` | Loneliness | Higher = more lonely | z-score | More lonely | Higher → risk | Strong cross-model driver |
| `aca_impa` | Academic impairment **due to** MH | Days/week academic work affected | z-score | More impairment | Higher → risk | **Outcome-proximal**; tautology concern |
| `flourish` | Flourishing scale | Higher = more flourishing | z-score | More flourishing | **Low** flourish → risk in SHAP | Protective construct |
| `positiveMH` | Positive mental health | Higher = better positive MH | z-score | Better positive MH | **Low** → risk | Outcome-proximal |
| `sub_any` | Any substance use | Higher = more use | z-score | More use | Higher → risk | Substance |
| `binge1pl` | Binge drinking | Higher = more binge | z-score | More binge | Higher → risk | Substance |
| `smok_cig` | Cigarette smoking | Higher = more smoking | z-score | More smoking | Higher → risk | Substance |
| `discrim_race` | Racial discrimination | Higher = more discrimination | z-score | More discrimination | Higher → risk | Stressor |
| `discrim_gender` | Gender discrimination | Higher = more | z-score | More | Higher → risk | Stressor |
| `discrim_sexual` | Sexual orientation discrimination | Higher = more | z-score | More | Higher → risk | Stressor |
| `inst_type` | Institution type | Coded category | z-score | Institution category | Context | Institution characteristic |
| `inst_size` | Institution size | Enrollment band | z-score | Larger | Context | Institution |
| `inst_public` | Public vs private | 1=public | z-score | Public | Context | Institution |

## SHAP direction vs raw coding

After **StandardScaler**, SHAP signs reflect association with elevated `dep_or_anx` in the **scaled** space:

- **Protective features** (`flourish`, `positiveMH`, `fincur`): low raw values often map to **positive SHAP** (increases predicted risk).
- **Risk features** (`lonely`, `aca_impa`, `food_worry`): high raw values map to **positive SHAP**.

Always cite the codebook (`Final 2024-25 Codebook (Student).pdf`) for exact survey wording and value labels.

## Target

| Variable | Definition |
|----------|------------|
| `dep_or_anx` | Binary screening flag: elevated depression and/or anxiety (1) vs not (0). **Not a clinical diagnosis.** |
