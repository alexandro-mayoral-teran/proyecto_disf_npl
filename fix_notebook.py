import json
import re

notebook_path = 'c:/bluepill5/Hack/Maestria/ProyectoIntegrador/proyecto_disf_npl/notebooks/Avance2__8_v4.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    modified = False

    # 1. Update Imports
    if 'from src.nlp_core.chunking import chunking_por_parrafo' in source:
        source = source.replace(
            'from src.nlp_core.chunking import chunking_por_parrafo, chunking_fijo_overlap, chunking_estructural, chunking_encabezados_md',
            'from src.nlp_core.chunking import RegulacionChunker, EstrategiaChunking'
        )
        modified = True

    # 2. Update Benchmark Dictionary
    if 'estrategias = {' in source and 'chunking_por_parrafo(texto)' in source:
        new_benchmark = '''    chunker_parrafo = RegulacionChunker(EstrategiaChunking.PARRAFO)
    chunker_fijo = RegulacionChunker(EstrategiaChunking.FIJO_OVERLAP, chunk_size=500, overlap=80)
    chunker_est = RegulacionChunker(EstrategiaChunking.ESTRUCTURAL)
    chunker_md = RegulacionChunker(EstrategiaChunking.ENCABEZADOS_MD, chunk_size=500, overlap=80)

    estrategias = {
        "parrafo": chunker_parrafo.chunk(texto),
        "fijo_overlap": chunker_fijo.chunk(texto),
        "estructural": chunker_est.chunk(texto),
        "encabezado_md": chunker_md.chunk(texto)
    }'''
        
        source = re.sub(r'estrategias\s*=\s*\{[^\}]+\}', new_benchmark, source)
        
        # Also update the len(c.split()) part
        source = source.replace('longitudes = [len(c.split()) for c in chunks]', 'longitudes = [len(c.page_content.split()) for c in chunks]')
        modified = True

    # 3. Update Dataset Creation
    if 'chunking_fijo_overlap(doc["texto_limpio"])' in source:
        source = source.replace('chunks_ml = chunking_fijo_overlap(doc["texto_limpio"])', 'chunker = RegulacionChunker(EstrategiaChunking.FIJO_OVERLAP)\n    chunks_ml = chunker.chunk(doc["texto_limpio"])')
        source = source.replace('chunks_fe = chunking_fijo_overlap(doc["texto_limpio"])', 'chunker = RegulacionChunker(EstrategiaChunking.FIJO_OVERLAP)\n    chunks_fe = chunker.chunk(doc["texto_limpio"])')
        source = source.replace('chunks_lid = chunking_fijo_overlap(doc["texto_limpio"])', 'chunker = RegulacionChunker(EstrategiaChunking.FIJO_OVERLAP)\n    chunks_lid = chunker.chunk(doc["texto_limpio"])')
        
        source = source.replace('"texto": chunk', '"texto": chunk.page_content')
        source = source.replace('len(chunk.split())', 'len(chunk.page_content.split())')
        modified = True

    if modified:
        # rebuild cell source list keeping newlines
        lines = source.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if len(lines) > 0 else []

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook updated successfully!')
