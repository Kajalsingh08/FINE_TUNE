import json
from typing import List, Dict
from pathlib import Path

class InstructionGenerator:
    """Generate instruction pairs for fine-tuning"""
    
    def __init__(self, metadata_path: str):
        with open(metadata_path) as f:
            self.metadata = json.load(f)
    
    def generate_instructions(self) -> List[Dict]:
        """Generate instruction pairs"""
        
        instructions = []
        cubes = self.metadata.get('cubes', [])
        
        for cube in cubes:
            cube_name = cube.get('name', 'Unknown')
            measures = cube.get('measures', [])
            dimensions = cube.get('dimensions', [])
            
            # Instruction 1: List measures
            if measures:
                instructions.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a metadata expert for Bayer Crop Science. Answer questions about cubes from your knowledge."
                        },
                        {
                            "role": "user",
                            "content": f"What measures are in {cube_name}?"
                        },
                        {
                            "role": "assistant",
                            "content": self._format_measures_answer(cube_name, measures)
                        }
                    ]
                })
            
            # Instruction 2: Primary key
            pk_dims = [d for d in dimensions if d.get('primaryKey')]
            if pk_dims:
                instructions.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a metadata expert for Bayer Crop Science. Answer questions about cubes from your knowledge."
                        },
                        {
                            "role": "user",
                            "content": f"What is the primary key of {cube_name}?"
                        },
                        {
                            "role": "assistant",
                            "content": f"The primary key of {cube_name} is {pk_dims[0].get('name')}, which is a {pk_dims[0].get('type')} dimension."
                        }
                    ]
                })
        
        return instructions
    
    def _format_measures_answer(self, cube_name: str, measures: List[Dict]) -> str:
        """Format measures as answer"""
        
        answer = f"The {cube_name} cube has {len(measures)} measures:\n\n"
        
        for i, measure in enumerate(measures, 1):
            m_name = measure.get('name', 'unknown')
            m_title = measure.get('title', m_name)
            agg_type = measure.get('aggType', 'unknown')
            
            answer += f"{i}. **{m_name}**\n"
            answer += f"   - Title: {m_title}\n"
            answer += f"   - Aggregation: {agg_type}\n\n"
        
        return answer.strip()
    
    def save_instructions(self, output_path: str):
        """Save instructions to JSON"""
        
        instructions = self.generate_instructions()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(instructions, f, indent=2)
        
        print(f"✓ Generated {len(instructions)} instruction pairs")
        print(f"✓ Saved to: {output_path}")

def main():
    generator = InstructionGenerator("./full_meta.json")
    generator.save_instructions("training_data/instructions_v1.json")


if __name__ == "__main__":
    main()