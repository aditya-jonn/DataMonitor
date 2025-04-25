# Limitations

**DataMonitor** is a tool for detecting deviations from known data distributions. However, its effectiveness has certain limitations that users should be aware of:

1. **Simulation-Based Validation**:  
   - DataMonitor was primarily tested and validated on simulated 'stepwise' data drift scenarios over time.  
   - This approach may not fully capture the complexity and unpredictability of real-world clinical data drifts.
   - The tool relies on simulations rather than true longitudinal datasets with timestamps, which may limit its applicability in clinical settings.

2. **Flagging Without Correction**:  
   - DataMonitor identifies and flags OOD inputs or data drifts.  
   - However, it does **not provide mechanisms** to:
     - Correct the detected drift.
     - Explain the underlying causes of the observed deviations.

