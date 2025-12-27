"""
Model Comparison Example

This example compares different survival analysis models supported by the framework
(PC-Hazard, LogisticHazard, DeepHit, DeepSurv, CoxPH, CoxTime, CoxCC).
"""
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter


def run_model(model_type, client_data, base_config):
    """Run federated learning with specific model type"""
    print(f"\n{'='*60}")
    print(f"Testing Model: {model_type}")
    print(f"{'='*60}")
    
    config = FSAConfig(
        **base_config,
        model_type=model_type,
        verbose=False  # Disable verbose for cleaner output
    )
    
    runner = FSARunner(config)
    results = runner.run(client_data, type='raw')
    
    print(f"Final Training C-index: {results['train_Cindex'][-1]:.4f}")
    print(f"Final Test C-index: {results['test_Cindex'][-1]:.4f}")
    print(f"Final Training IBS: {results['train_IBS'][-1]:.4f}")
    print(f"Final Test IBS: {results['test_IBS'][-1]:.4f}")
    
    return results


def main():
    """Main function: Compare different survival analysis models"""
    
    print("=== Survival Analysis Model Comparison ===\n")
    
    # Generate and partition data
    print("Step 1: Generating and partitioning data...")
    sim_config = SimulationConfig(
        n_samples=500,
        n_features=10,
        random_state=42
    )
    
    generator = DataGenerator(sim_config)
    data = generator.generate('weibull')
    
    splitter = DataSplitter(
        n_clients=3,
        split_type='iid',
        test_size=0.2,
        random_state=42
    )
    client_data = splitter.split(data)
    print(f"  Data split into {len(client_data.clients_set)} clients")
    
    # Base configuration
    base_config = {
        'num_clients': 3,
        'n_features': 10,
        'n_samples': 500,
        'global_epochs': 10,
        'local_epochs': 2,
        'learning_rate': 0.01,
        'random_seed': 42,
    }
    
    print("\nStep 2: Training models...")
    
    # Define models to test
    models = [
        ('PC-Hazard', 'Piecewise constant hazard model'),
        ('LogisticHazard', 'Logistic regression-based hazard model'),
        ('DeepHit', 'Deep learning-based survival model'),
        ('DeepSurv', 'Deep learning proportional hazards model'),
        ('CoxPH', 'Traditional Cox proportional hazards'),
        ('CoxTime', 'Time-dependent Cox model'),
        ('CoxCC', 'Case-control Cox model'),
    ]
    
    results_dict = {}
    
    for model_type, description in models:
        print(f"\nDescription: {description}")
        try:
            results = run_model(model_type, client_data, base_config)
            results_dict[model_type] = results
        except Exception as e:
            print(f"  Error running {model_type}: {e}")
            continue
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY: Model Performance Comparison")
    print("="*60)
    print(f"\n{'Model':<20} {'Train C-index':<15} {'Test C-index':<15} {'Train IBS':<12} {'Test IBS':<12}")
    print("-"*80)
    
    for model_type, results in results_dict.items():
        train_cindex = results['train_Cindex'][-1]
        test_cindex = results['test_Cindex'][-1]
        train_ibs = results['train_IBS'][-1]
        test_ibs = results['test_IBS'][-1]
        print(f"{model_type:<20} {train_cindex:<15.4f} {test_cindex:<15.4f} {train_ibs:<12.4f} {test_ibs:<12.4f}")
    
    # Model characteristics
    print("\n" + "="*60)
    print("Model Characteristics")
    print("="*60)
    
    print("\n1. PC-Hazard")
    print("   - Piecewise constant hazard model")
    print("   - Discretizes time into intervals")
    print("   - Uses quantile-based time discretization")
    print("   - Good for general survival analysis")
    
    print("\n2. LogisticHazard")
    print("   - Similar to PC-Hazard with logistic activation")
    print("   - Better for smooth hazard functions")
    print("   - Uses quantile-based time discretization")
    
    print("\n3. DeepHit")
    print("   - Deep learning-based model")
    print("   - Captures complex non-linear relationships")
    print("   - Can handle competing risks")
    print("   - Uses quantile-based time discretization")
    
    print("\n4. DeepSurv")
    print("   - Deep learning proportional hazards")
    print("   - No time discretization needed")
    print("   - Good baseline model")
    
    print("\n5. CoxPH")
    print("   - Traditional Cox proportional hazards")
    print("   - Assumes proportional hazards")
    print("   - No time discretization needed")
    
    print("\n6. CoxTime")
    print("   - Time-dependent Cox model")
    print("   - More flexible than CoxPH")
    print("   - Allows time-varying effects")
    
    print("\n7. CoxCC")
    print("   - Case-control Cox model")
    print("   - Efficient for large datasets")
    print("   - Suitable for matched case-control studies")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
