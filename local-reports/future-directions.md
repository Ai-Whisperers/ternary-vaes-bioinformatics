# What kind of network can provide *correct feedback* to StateNet with both VAEs?

First we need to verify this affirmation: **NOW**, exactly as it is implemented today, **StateNet is 100% feed-forward**, with:

* no memory
* no recurrence
* no temporal dynamics
* no RNN/GRU/LSTM
* no ODE structure
* no internal state besides the input at each iteration

This has advantages (stability, simplicity, reproducibility),
but also one big limitation:

> **If the network cannot *anticipate*. It only *reacts*.**

We want StateNet to receive correct (deep, temporal, dynamic) feedback from **both VAEs**, the system needs:

* true temporal memory
* feedback dynamics
* tracking of long-term drift
* detection of oscillation patterns
* momentum over KL and entropy
* anticipation of instability
* smooth control of the coupling ρ

The only way to achieve that is to give the controller **a network that models temporal dynamics**, not just instantaneous values.

Our architecture is *perfectly* suited for adding a **dynamic feedback subsystem**, not a representational network.

Below are the correct architectural options, from simplest to most advanced:





Dime que escoges entre estas opciones, ya que hay varias rutas a partir de ahora:

0. Verificar la naturaleza tecnica computacional detras de la definicion de "ternary manifold", si todo el sistema esta generando el espacio ternario a partir de "alucinaciones" de lo discreto, estamos simplemente dibujando un espacio latente que es util, pero no es revolucionario (ni practico, ya que no es el manifoldio grupoide de operaciones ternarias)

1. Aislar ambos VAEs, al vae A para estudiar sus propiedades con mas detalle, que funciones prefiere y que "parte" del manifold ternario mapea mejor (pero con definicion formal y benchmarkeada), al vae B para estudiar lo mismo. Luego probar que ocurre en combinaciones como: VAE-B + StateNet (sin ningun VAE-A) y viceversa (VAE-A + StateNet, sin ningun VAE-B). Es decir esto si no lo probamos, revisar que ocurre cuando el statenet igual influencia al VAE huerfano. Esto podria revelar mucho de las naturalezas de todo el sistema complejo, tanto de el sistema de aprendizaje como del sistema informacional que se busca entender (el ternary manifold)

2. Implementar ya mismo mejoras y optimizaciones no disruptivas para asegurar aun mas la calidad de los benchmarks y no desperdiciar computo en epochs de training

3. Implementar ya mismo mejoras y optimizaciones altamente disruptivas para asegurar exploracion valiosa y nunca antes documentada