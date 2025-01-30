from RAGPipeline import RAGPiepeline
from dotenv import load_dotenv
load_dotenv()

system_prompt = ('Sei un assistente virtuale in ambito manifatturiero e devi rispondere alle domande in input nel '
                 'modo più completo possibile basandoti solo sul contesto. Limitati a rispondere alla domande che ti '
                 'viene posta, non generare altre domande!')
pipeline = RAGPiepeline(system_prompt=system_prompt)
answer = pipeline.run(user_question='Qual è la temperatura operativa del Forno WB?')
print(answer)