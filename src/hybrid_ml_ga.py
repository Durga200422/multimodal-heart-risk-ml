#!/usr/bin/env python3
"""
Hybrid ML+GA Model: Genetic Algorithm for Ensemble Weight Optimization

This approach uses a Genetic Algorithm to evolve optimal weights for combining
predictions from multiple ML models. Unlike traditional stacking (which uses
a meta-learner), GA directly optimizes the weighted combination.

Algorithm:
1. Train multiple base ML models (RF, XGB, LGB, SVM)
2. Use GA to evolve optimal weights for combining their predictions
3. Fitness function: ROC-AUC score on validation set
4. Final prediction: weighted average of base model probabilities

Benefits:
- No additional training of meta-learner needed
- Interpretable weights show contribution of each base model
- GA explores non-linear optimization space efficiently
- Prevents overfitting through cross-validation fitness
"""

from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_prep import load_scenario_a, load_scenario_b
from src.eval_utils import (
    compute_classification_metrics,
    save_metrics_table,
    plot_confusion,
)

warnings.filterwarnings('ignore')

RESULTS_DIR_A = Path(__file__).resolve().parents[1] / "results" / "scenario_a"
RESULTS_DIR_B = Path(__file__).resolve().parents[1] / "results" / "scenario_b"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing ensemble weights.
    
    Chromosome: Array of weights for each base model
    Fitness: ROC-AUC score of weighted ensemble predictions
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        verbose: int = 1,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.verbose = verbose
        
        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.fitness_history = []
    
    def _initialize_population(self, n_models: int) -> np.ndarray:
        """
        Initialize population with random weights.
        Each chromosome is normalized to sum to 1.
        """
        population = np.random.random((self.population_size, n_models))
        # Normalize each chromosome
        population = population / population.sum(axis=1, keepdims=True)
        return population
    
    def _fitness(
        self,
        chromosome: np.ndarray,
        predictions: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Calculate fitness of a chromosome.
        
        Args:
            chromosome: Weights for each model
            predictions: Array of shape (n_samples, n_models) with model predictions
            y_true: True labels
            
        Returns:
            ROC-AUC score
        """
        # Weighted average of predictions
        weighted_pred = np.dot(predictions, chromosome)
        
        # Clip to valid probability range
        weighted_pred = np.clip(weighted_pred, 1e-7, 1 - 1e-7)
        
        # Calculate ROC-AUC
        try:
            score = roc_auc_score(y_true, weighted_pred)
        except:
            score = 0.0
        
        return score
    
    def _evaluate_population(
        self,
        population: np.ndarray,
        predictions: np.ndarray,
        y_true: np.ndarray
    ) -> np.ndarray:
        """Evaluate fitness for entire population."""
        fitness_scores = np.array([
            self._fitness(chromosome, predictions, y_true)
            for chromosome in population
        ])
        return fitness_scores
    
    def _selection(
        self,
        population: np.ndarray,
        fitness_scores: np.ndarray
) -> np.ndarray: # Removed the second Tuple return type as it wasn't used
        """
        Tournament selection: randomly select k individuals and pick the best.
        """
        tournament_size = 3
        selected = []
    
        # Use the actual length of the incoming population/fitness_scores
        current_pop_size = len(fitness_scores) 
    
        for _ in range(self.population_size - self.elite_size):
            # Random tournament using the current size
            indices = np.random.choice(
                current_pop_size, 
                size=tournament_size,
                replace=False
            )
            tournament_fitness = fitness_scores[indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return np.array(selected)
    
    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniform crossover: each gene has 50% chance from either parent.
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        mask = np.random.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        
        # Normalize to sum to 1
        child1 = child1 / child1.sum()
        child2 = child2 / child2.sum()
        
        return child1, child2
    
    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Gaussian mutation: add small random noise to genes.
        """
        if np.random.random() > self.mutation_rate:
            return chromosome
        
        # Add Gaussian noise
        noise = np.random.normal(0, 0.1, size=chromosome.shape)
        mutated = chromosome + noise
        
        # Ensure non-negative and normalize
        mutated = np.abs(mutated)
        mutated = mutated / mutated.sum()
        
        return mutated
    
    def evolve(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray
    ) -> np.ndarray:
        """
        Run the genetic algorithm.
        
        Args:
            predictions: Array of shape (n_samples, n_models)
            y_true: True labels
            
        Returns:
            Best weights found
        """
        n_models = predictions.shape[1]
        
        # Initialize population
        population = self._initialize_population(n_models)
        
        if self.verbose > 0:
            print(f"\n🧬 Running Genetic Algorithm...")
            print(f"  Population size: {self.population_size}")
            print(f"  Generations: {self.generations}")
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = self._evaluate_population(population, predictions, y_true)
            
            # Track best
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_chromosome = population[best_idx].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            if self.verbose > 0 and (generation + 1) % 20 == 0:
                avg_fitness = fitness_scores.mean()
                print(f"  Gen {generation+1}/{self.generations} | "
                      f"Best: {self.best_fitness:.4f} | "
                      f"Avg: {avg_fitness:.4f}")
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elite = population[elite_indices]
            
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                parent1, parent2 = selected[i], selected[i + 1]
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                offspring.extend([child1, child2])
            
            # Combine elite and offspring for next generation
            population = np.vstack([elite, offspring[:len(selected)]])
        
        if self.verbose > 0:
            print(f"✅ GA Complete! Best fitness: {self.best_fitness:.4f}")
        
        return self.best_chromosome


class HybridMLGA:
    """
    Hybrid ML+GA ensemble using Genetic Algorithm for weight optimization.
    """
    
    def __init__(self, scenario: str = "a", verbose: int = 1):
        self.scenario = scenario
        self.verbose = verbose
        self.base_models = {}
        self.optimal_weights = None
        self.ga = None
    
    def _build_base_models(self) -> Dict:
        """Build base ML models."""
        if self.scenario == "a":
            models = {
                "rf": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced",
                ),
                "xgb": XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=42,
                ),
                "lgb": LGBMClassifier(
                    n_estimators=300,
                    max_depth=-1,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1,
                ),
                "svm": LinearSVC(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                ),
            }
        else:
            models = {
                "rf": RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced",
                ),
                "xgb": XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=42,
                ),
                "lgb": LGBMClassifier(
                    n_estimators=400,
                    max_depth=-1,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1,
                ),
                "svm": SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                ),
            }
        return models
    
    def _get_predictions(
        self,
        X: np.ndarray,
        fit: bool = False,
        y: np.ndarray = None
    ) -> np.ndarray:
        """Get probability predictions from all base models."""
        predictions = []
        
        for name, model in self.base_models.items():
            if fit:
                if self.verbose > 0:
                    print(f"  Training: {name}")
                model.fit(X, y)
            
            if name == "svm" and isinstance(model, LinearSVC):
                scores = model.decision_function(X)
                proba = 1 / (1 + np.exp(-scores))
            else:
                proba = model.predict_proba(X)[:, 1]
            
            predictions.append(proba)
        
        return np.column_stack(predictions)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        ga_params: Dict = None,
    ):
        """
        Train hybrid ML+GA model.
        
        1. Train base ML models on training data
        2. Use GA to optimize weights on validation data
        """
        if self.verbose > 0:
            print(f"\n🔷 Training Hybrid ML+GA for Scenario {self.scenario.upper()}")
            print("Stage 1: Training base ML models...")
        
        # Train base models
        self.base_models = self._build_base_models()
        train_preds = self._get_predictions(X_train, fit=True, y=y_train)
        
        # Get validation predictions
        val_preds = self._get_predictions(X_val, fit=False)
        
        if self.verbose > 0:
            print("\nStage 2: Optimizing ensemble weights with GA...")
        
        # Initialize GA
        ga_params = ga_params or {}
        self.ga = GeneticAlgorithm(verbose=self.verbose, **ga_params)
        
        # Optimize weights on validation set
        self.optimal_weights = self.ga.evolve(val_preds, y_val)
        
        if self.verbose > 0:
            print("\n📊 Optimal ensemble weights:")
            for name, weight in zip(self.base_models.keys(), self.optimal_weights):
                print(f"  {name}: {weight:.4f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        predictions = self._get_predictions(X, fit=False)
        weighted_pred = np.dot(predictions, self.optimal_weights)
        return np.clip(weighted_pred, 1e-7, 1 - 1e-7)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def save(self, path: Path):
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'scenario': self.scenario,
            'base_models': self.base_models,
            'optimal_weights': self.optimal_weights,
            'ga_fitness_history': self.ga.fitness_history if self.ga else None,
        }
        
        joblib.dump(model_data, path)
        
        if self.verbose > 0:
            print(f"💾 Model saved to: {path}")
    
    @classmethod
    def load(cls, path: Path, verbose: int = 1):
        """Load saved model."""
        model_data = joblib.load(path)
        
        model = cls(scenario=model_data['scenario'], verbose=verbose)
        model.base_models = model_data['base_models']
        model.optimal_weights = model_data['optimal_weights']
        
        if verbose > 0:
            print(f"📂 Model loaded from: {path}")
        
        return model


def train_and_evaluate_ga_scenario_a():
    """Train and evaluate Hybrid ML+GA for Scenario A."""
    print("\n" + "="*70)
    print("HYBRID ML+GA - SCENARIO A (Lifestyle)")
    print("="*70)
    
    data = load_scenario_a()
    X_train, X_test = data.X_train, data.X_test
    y_train, y_test = data.y_train, data.y_test
    
    # Validation split
    val_size = int(0.2 * len(X_train))
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train_sub = X_train[val_size:]
    y_train_sub = y_train[val_size:]
    
    # Train
    hybrid = HybridMLGA(scenario="a", verbose=1)
    ga_params = {
        'population_size': 50,
        'generations': 100,
        'mutation_rate': 0.15,
        'crossover_rate': 0.7,
        'elite_size': 5,
    }
    hybrid.fit(X_train_sub, y_train_sub, X_val, y_val, ga_params=ga_params)
    
    # Evaluate
    y_proba = hybrid.predict_proba(X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    
    print("\n📊 Hybrid ML+GA Metrics (Scenario A):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save
    RESULTS_DIR_A.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "hybrid_ml_ga_scenario_a.pkl"
    hybrid.save(model_path)
    
    metrics_dict = {"HybridMLGA": metrics}
    metrics_csv = RESULTS_DIR_A / "hybrid_ml_ga_metrics.csv"
    save_metrics_table(metrics_dict, str(metrics_csv))
    
    cm_path = RESULTS_DIR_A / "confusion_HybridMLGA.png"
    plot_confusion(y_test, y_pred, str(cm_path),
                   "Scenario A - Hybrid ML+GA Confusion Matrix")
    
    return hybrid, metrics


def train_and_evaluate_ga_scenario_b():
    """Train and evaluate Hybrid ML+GA for Scenario B."""
    print("\n" + "="*70)
    print("HYBRID ML+GA - SCENARIO B (Clinical)")
    print("="*70)
    
    data = load_scenario_b()
    X_train, X_test = data.X_train, data.X_test
    y_train, y_test = data.y_train, data.y_test
    
    # Validation split
    val_size = int(0.2 * len(X_train))
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train_sub = X_train[val_size:]
    y_train_sub = y_train[val_size:]
    
    # Train
    hybrid = HybridMLGA(scenario="b", verbose=1)
    ga_params = {
        'population_size': 50,
        'generations': 100,
        'mutation_rate': 0.15,
        'crossover_rate': 0.7,
        'elite_size': 5,
    }
    hybrid.fit(X_train_sub, y_train_sub, X_val, y_val, ga_params=ga_params)
    
    # Evaluate
    y_proba = hybrid.predict_proba(X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    
    print("\n📊 Hybrid ML+GA Metrics (Scenario B):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save
    RESULTS_DIR_B.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "hybrid_ml_ga_scenario_b.pkl"
    hybrid.save(model_path)
    
    metrics_dict = {"HybridMLGA": metrics}
    metrics_csv = RESULTS_DIR_B / "hybrid_ml_ga_metrics.csv"
    save_metrics_table(metrics_dict, str(metrics_csv))
    
    cm_path = RESULTS_DIR_B / "confusion_HybridMLGA.png"
    plot_confusion(y_test, y_pred, str(cm_path),
                   "Scenario B - Hybrid ML+GA Confusion Matrix")
    
    return hybrid, metrics


if __name__ == "__main__":
    np.random.seed(42)
    
    RESULTS_DIR_A.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR_B.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train both scenarios
    hybrid_a, metrics_a = train_and_evaluate_ga_scenario_a()
    hybrid_b, metrics_b = train_and_evaluate_ga_scenario_b()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Scenario A ROC-AUC: {metrics_a['roc_auc']:.4f}")
    print(f"Scenario B ROC-AUC: {metrics_b['roc_auc']:.4f}")