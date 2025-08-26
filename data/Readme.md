# Dataset Repository

This repository organizes the datasets used for development, testing, and reference in our experiments.  
It is not intended for running code directly, but rather as a **data source** for other modules.

---

## Structure

- **`dev data 1 (for prompting)`**  
  Contains development data primarily used for building prompts and initial trials.

- **`dev data 2 (for development)`**  
  Contains development data reserved for model development and hyperparameter tuning.

- **`full data (only for storing, do not use)`**  
  Complete dataset kept only for storage and archival purposes.  
  Not to be used directly in experiments.

- **`general data (some additional stuff)`**  
  Miscellaneous supporting datasets and auxiliary files.

- **`original papers reference data (leakage checks)`**  
  Reference data drawn from original publications.  
  Useful for cross-checking leakage and evaluation consistency.

- **`test data (only use at the end)`**  
  Held-out final test set.  
  Not used during development or tuning — for final evaluation only.

---

## Notes
- Use **Dev Data 1** for prompt design and early exploration.  
- Use **Dev Data 2** for iterative testing.  
- Use **Test Data** strictly for final evaluation to ensure unbiased results.  
- The **Full Data** folder is provided only for completeness and storage.

---
