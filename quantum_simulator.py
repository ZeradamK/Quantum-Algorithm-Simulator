from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import circuit_drawer
from qiskit_optimization.applications import Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.applications import Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.primitives import Estimator
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

class QuantumSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits, num_qubits)

    def apply_hadamard(self):
        """Apply Hadamard gate to all qubits."""
        for qubit in range(self.num_qubits):
            self.circuit.h(qubit)

    def apply_custom_gate(self, gate_type):
        """Apply a custom gate (X, Z) to all qubits."""
        if gate_type.lower() == "x":
            for qubit in range(self.num_qubits):
                self.circuit.x(qubit)
        elif gate_type.lower() == "z":
            for qubit in range(self.num_qubits):
                self.circuit.z(qubit)
        else:
            print(f"Unknown gate type: {gate_type}. Skipping...")

    def measure(self):
        """Add measurement to all qubits."""
        self.circuit.measure(range(self.num_qubits), range(self.num_qubits))

    def simulate(self):
        """Simulate the quantum circuit."""
        simulator = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(self.circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=1024)
        result = job.result()
        return result.get_counts()

    def visualize(self):
        """Visualize the quantum circuit in text format."""
        return self.circuit.draw('text')

    def solve_tsp(self, num_cities):
        """Solve the Traveling Salesman Problem using QAOA."""
        
        tsp = Tsp.create_random_instance(num_cities, seed=42)
        qp = tsp.to_quadratic_program()

        
        qubo_converter = QuadraticProgramToQubo()
        qubo = qubo_converter.convert(qp)

       
        algorithm_globals.random_seed = 42
        estimator = Estimator()  
        optimizer = COBYLA(maxiter=100)  
        qaoa = QAOA(estimator, optimizer=optimizer, reps=1, initial_point=[0.0, 0.0])

       
        eigen_optimizer = MinimumEigenOptimizer(qaoa)
        result = eigen_optimizer.solve(qubo)

      
        solution = tsp.interpret(result)

      
        qc = qaoa.construct_circuit(result.x).decompose()
        return solution, tsp, qc



    def visualize_circuit(self, qc, filename='tsp_circuit.png'):
        """Visualize and save the quantum circuit."""
        circuit_drawer(qc, output='mpl', filename=filename)
        print(f"Circuit diagram saved as {filename}")


if __name__ == "__main__":
    
    while True:
        try:
            num_cities = int(input("Enter the number of cities for the TSP simulation: "))
            if num_cities < 2:
                print("Please enter a number greater than 1.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    sim = QuantumSimulator(num_qubits=num_cities)
    solution, tsp_instance, qc = sim.solve_tsp(num_cities)

    print(f"Optimal route: {solution}")
    sim.visualize_circuit(qc)
