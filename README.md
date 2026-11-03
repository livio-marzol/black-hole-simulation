# 🌌 Gargantua — Simulador Relativístico de Buraco Negro (Three.js + GLSL)

Simulação interativa 3D em tempo real de um **Buraco Negro de Schwarzschild** rodando diretamente no navegador via **WebGL**, **Three.js** e **GLSL Shaders** customizados.

## ✨ Recursos Físicos e Gráficos

- **Raymarching Geodésico (Relatividade Geral):** Integração numérica da trajetória dos fótons na Métrica de Schwarzschild, gerando o **Horizonte de Eventos**, a **Esfera de Fótons ($1.5 R_s$)** e a **Lente Gravitacional** do campo estelar e do disco.
- **Disco de Acreção Relativístico:** Rotação diferencial kepleriana com filamentos de plasma, **Efeito Doppler Relativístico (*Doppler Beaming*)** e **Redshift Gravitacional**.
- **Mecânica Orbital & Espaguetificação:** Corpos celestes (asteroides, cometas e estrelas) orbitam sob o potencial relativístico de **Paczyński–Wiita**, sofrendo deformação por tensor de maré (**espaguetificação**), aquecimento incandescente e **Eventos de Ruptura de Maré (TDE)** ao cruzarem a órbita circular estável mínima (**ISCO = $3 R_s$**).
- **Performance Adaptativa (Desktop & Mobile):** Governador dinâmico de resolução e passos que mantém 60 FPS em GPUs integradas (AMD Vega 7) e celulares (Qualcomm Adreno 610).

## 🚀 Como Executar Localmente

Basta servir a pasta com qualquer servidor HTTP estático:

```bash
npx http-server . -p 8090
```

Depois acesse `http://localhost:8090` no navegador.
