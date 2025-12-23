"""
Instance file generator for the manufacturing workshop.
Generates JSS instance files based on part quantities.
"""

from pathlib import Path
from typing import Tuple


class WorkshopInstanceGenerator:
    """Generate JSS instance files for the manufacturing workshop."""
    
    # Machine indices
    MACHINES = {
        'CNC': 0,
        'Fraiseuse': 1,
        'Tour': 2,
        'Perceuse': 3,
        'Polisseuse': 4
    }
    
    # Part definitions: (name, operations sequence)
    # Operations: [(machine, duration), ...]
    # Chaque pièce passe maintenant par TOUTES les machines (5 opérations)
    PARTS = {
        'A': {
            'name': 'Support métallique',
            'operations': [
                ('CNC', 8),
                ('Fraiseuse', 5),
                ('Tour', 2),         # Ajouté: ébavurage sur tour
                ('Perceuse', 3),
                ('Polisseuse', 2)
            ]
        },
        'B': {
            'name': 'Axe cylindrique',
            'operations': [
                ('Tour', 7),
                ('CNC', 6),
                ('Fraiseuse', 3),    # Ajouté: finition sur fraiseuse
                ('Perceuse', 2),     # Ajouté: perçage de trous de fixation
                ('Polisseuse', 3)
            ]
        },
        'C': {
            'name': 'Plaque percée',
            'operations': [
                ('Fraiseuse', 6),    # Commence par fraisage des bords
                ('Perceuse', 4),     # Perçage des trous
                ('CNC', 5),          # Usinage de précision
                ('Tour', 3),         # Chanfreinage
                ('Polisseuse', 4)    # Finition
            ]
        }
    }
    
    def __init__(self):
        self.num_machines = len(self.MACHINES)
    
    def generate_instance(self, num_a: int, num_b: int, num_c: int, 
                         output_path: str = None) -> Tuple[str, dict]:
        """
        Generate a JSS instance file based on part quantities.
        
        Args:
            num_a: Number of part A to produce
            num_b: Number of part B to produce
            num_c: Number of part C to produce
            output_path: Path to save the instance file (optional)
            
        Returns:
            Tuple of (instance_path, instance_info)
        """
        # Calculate total jobs
        total_jobs = num_a + num_b + num_c
        
        if total_jobs == 0:
            raise ValueError("At least one part must be ordered")
        
        # Build instance content
        lines = []
        
        # Header: number of jobs and machines
        lines.append(f"{total_jobs} {self.num_machines}")
        
        # Add jobs for each part
        job_descriptions = []
        
        # Part A jobs
        for i in range(num_a):
            job_line, max_ops = self._generate_job_line('A')
            lines.append(job_line)
            job_descriptions.append(f"Job {len(job_descriptions)}: Part A #{i+1}")
        
        # Part B jobs
        for i in range(num_b):
            job_line, max_ops = self._generate_job_line('B')
            lines.append(job_line)
            job_descriptions.append(f"Job {len(job_descriptions)}: Part B #{i+1}")
        
        # Part C jobs
        for i in range(num_c):
            job_line, max_ops = self._generate_job_line('C')
            lines.append(job_line)
            job_descriptions.append(f"Job {len(job_descriptions)}: Part C #{i+1}")
        
        instance_content = '\n'.join(lines)
        
        # Save to file if path provided
        if output_path is None:
            output_path = f"instances/workshop_{num_a}A_{num_b}B_{num_c}C.txt"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(instance_content)
        
        # Prepare instance info
        instance_info = {
            'path': str(output_file),
            'num_a': num_a,
            'num_b': num_b,
            'num_c': num_c,
            'total_jobs': total_jobs,
            'num_machines': self.num_machines,
            'job_descriptions': job_descriptions,
            'estimated_min_makespan': self._estimate_min_makespan(num_a, num_b, num_c)
        }
        
        return str(output_file), instance_info
    
    def _generate_job_line(self, part_type: str) -> Tuple[str, int]:
        """Generate a job line for the instance file."""
        part = self.PARTS[part_type]
        operations = part['operations']
        
        # Vérifier que chaque pièce a exactement num_machines opérations
        if len(operations) != self.num_machines:
            raise ValueError(f"Part {part_type} doit avoir exactement {self.num_machines} opérations, "
                           f"a {len(operations)}")
        
        # Convert operations to machine_index duration pairs
        job_data = []
        for machine_name, duration in operations:
            machine_idx = self.MACHINES[machine_name]
            job_data.append(f"{machine_idx} {duration}")
        
        return '  '.join(job_data), len(operations)
    
    def _estimate_min_makespan(self, num_a: int, num_b: int, num_c: int) -> int:
        """
        Estimate theoretical minimum makespan (lower bound).
        This assumes perfect parallelization, which is usually impossible.
        """
        # Calculate total time needed on each machine
        machine_times = [0] * self.num_machines
        
        for part_type, quantity in [('A', num_a), ('B', num_b), ('C', num_c)]:
            operations = self.PARTS[part_type]['operations']
            for machine_name, duration in operations:
                machine_idx = self.MACHINES[machine_name]
                machine_times[machine_idx] += duration * quantity
        
        # The bottleneck machine determines the minimum makespan
        return max(machine_times)
    
    def display_instance_info(self, instance_info: dict):
        """Display instance information in a readable format."""
        print("=" * 70)
        print("INSTANCE GÉNÉRÉE")
        print("=" * 70)
        print(f"Fichier: {instance_info['path']}")
        print(f"\nCommande:")
        print(f"  - Pièce A (Support métallique):  {instance_info['num_a']} unités")
        print(f"  - Pièce B (Axe cylindrique):     {instance_info['num_b']} unités")
        print(f"  - Pièce C (Plaque percée):       {instance_info['num_c']} unités")
        print(f"\nTotal: {instance_info['total_jobs']} jobs sur {instance_info['num_machines']} machines")
        print(f"Makespan minimum théorique: {instance_info['estimated_min_makespan']} minutes")
        print(f"  (avec parallélisation parfaite - généralement impossible)")
        print("=" * 70)


if __name__ == "__main__":
    # Test the generator
    generator = WorkshopInstanceGenerator()
    
    print("Test 1: Small order")
    instance_path, info = generator.generate_instance(2, 1, 1)
    generator.display_instance_info(info)
    
    print("\n" + "="*70 + "\n")
    
    print("Test 2: Larger order")
    instance_path, info = generator.generate_instance(5, 3, 4)
    generator.display_instance_info(info)
    
    print("\n" + "="*70)
    print(f"\nInstance files created in: instances/")
