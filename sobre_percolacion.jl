### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ ba216328-fa4c-11eb-0c08-fbbe22b707ab
using GraphRecipes

# ╔═╡ 707c6828-fa4a-11eb-31a6-19b6713fd616
using LightGraphs

# ╔═╡ 108fa28a-fab9-11eb-0ab8-1b971c2c0dbf
using PyCall

# ╔═╡ df1964f2-f635-11eb-22b9-d987e3d2e5be
using Plots

# ╔═╡ 1491f03a-fa47-11eb-023d-5fea359ab4a8
using SciPy

# ╔═╡ 8dc882cc-fa4f-11eb-23d9-1d852138ab04
using PlutoUI

# ╔═╡ 8e6d427c-f62a-11eb-131e-3357d8fad432
md"# Sobre Teoría de Percolación"

# ╔═╡ 91ebd6c0-fa4e-11eb-1453-e3217e276dfb
md"""### Modelo de Barabasi-Albert"""

# ╔═╡ c2344572-fabe-11eb-1049-651fc12fe17b
PlutoUI.TableOfContents(aside=true, title="Sobre Teoría de Percolación")

# ╔═╡ e1c28daa-fab3-11eb-187d-3b4cb5788abb
md"""
Comenzamos con ``m_0`` vértices, las aristas entre ellos son elegidas arbitrariamente, siempre que cada vértice tiene al menos una arista. La red se desarrolla siguiendo los siguientes dos pasos:

a) Crecimiento: En cada paso de tiempo, agregamos un nuevo vértice con ``m (≤ m0)`` aristas que conectan el nuevo vértice con ``m`` vértices que ya están en la red.

b) Enlace preferencial: La probabilidad ``Π (k)`` de que un enlace del nuevo vértice se conecte al vértice ``i`` depende del grado ``k_i`` siguiendo ``\Pi(k_i) = \frac{k_i}{\sum_{j} k_j}.``

El enlace preferencial es un mecanismo probabilístico: un nuevo vértice puede conectarse libremente a cualquier vértice de la red, ya sea un hub o a un enlace único. La ecuación anbterior implica que si un nuevo vértice puede elegir entre un vértice de grado dos y uno de grado cuatro, es dos veces más probable que se conecte al vértice de grado cuatro.
"""

# ╔═╡ 5ceaba4c-fab5-11eb-1935-4518850730f6
md"""
La siguiente es una función del paquetre LightGraphs que genera el modelo. Esta crea una gráfica aleatoria del modelo de Barabási-Albert con ``n`` vértices. Se hace crecer agregando nuevos vértices a uà gráfica inicial con ``n0`` vértices. Cada nuevo vértice se adjunta con ``k`` aristas a ``k`` diferentes vértices ya presentes en el sistema mediante un enlace preferencial. Las gráficas iniciales no están dirigidas y constan de vértices aislados de forma predeterminada.
"""

# ╔═╡ 5611c6cc-fa4f-11eb-3521-0f0d2fd67f72
@bind n0 Slider(5:10)

# ╔═╡ d355d844-fa4f-11eb-2f3b-496ab03ce67b
begin
g = barabasi_albert(12, n0,5)
graphplot(g, curves=false)
end

# ╔═╡ d4783d24-fabe-11eb-1a65-a96571209d30
md"""### Transición de fase en el modelo de BA"""

# ╔═╡ e2afe1a6-fab6-11eb-08df-9baeedf9317a
md"""
Una pregunta natural que podemos hacernos es: ¿cuándo está conectada ``BA(n,n_0,k)``? Para ello usaremos la función is\_connected para verificar en varios valores de ``n_0`` y ``k`` en qué momento toda la red se encuentra conectada. Esto corresponde con un cambio de fase en el modelo que representa esta red. 

Nótese que por las propiedades de Pluto, al variar la gráfica g, el resultado de la función irá variando.  

"""

# ╔═╡ 7b6b265e-fab6-11eb-3993-a1ea0ebe7a16
is_connected(g)

# ╔═╡ 176be4cc-fabb-11eb-05a4-db6723be5ed3
md"""Obtenemos las componentes conexas de g y luego el tamaño de la más grande"""

# ╔═╡ 22924a36-faba-11eb-2b50-c74fcf87051d
c_comp = connected_components(g)

# ╔═╡ 7f3f3dd4-faba-11eb-3b10-bdf9961ffda1
maximum(length.(c_comp))

# ╔═╡ 855e718e-fabb-11eb-050b-718ddc7256e7
md"""Generamos ahora 100 gráficas BA y obtenemos de cada una de ella el tamaño de la componente conexa más grande. Comenzaremos primero variando el valor ``n_0``y luego el valor de ``k``."""

# ╔═╡ 20e16c34-fab9-11eb-1fd2-7599e4e4e1c9
np = pyimport("numpy")

# ╔═╡ 5eedfc0a-fab7-11eb-11e1-77aec110c445
n0_array = np.arange(10, 1010,10 )

# ╔═╡ 4f80321c-fab8-11eb-2581-8989e71dd9a2
connected_n0=[]

# ╔═╡ 2f7ffae4-fab8-11eb-2e7b-11dc13b26851
for n0 in n0_array
	h = barabasi_albert(1010, n0,10)
	c_comp = connected_components(h)
	connected = maximum(length.(c_comp))
	push!(connected_n0,connected)
end

# ╔═╡ 5bc5c400-fab9-11eb-114d-312f9139ae71
begin
plot(n0_array,connected_n0)
plot!(title="Transición de fase de la conexidad de una gráfica BA")
plot!(ylabel = "Tamaño de la componente conexa más grande")
xlabel!("n_0")
end

# ╔═╡ 6ade9dee-fabc-11eb-07c6-8fde2683c9e7
k_array = np.arange(10,100,1 )

# ╔═╡ 48a0deac-fabc-11eb-242e-eb0df0411472
connected_k=[]

# ╔═╡ 443490b6-fabc-11eb-3028-956a647500a7
for k in k_array 
	h1 = barabasi_albert(1000, 110 ,k)
	c_comp = connected_components(h1)
	connected = maximum(length.(c_comp))
	push!(connected_k,connected)
end

# ╔═╡ a36620f4-fabc-11eb-1fbd-59a5d323dada
begin
plot(k_array,connected_k)
plot!(legend=:bottomright)
plot!(title="Transición de fase de la conexidad de una gráfica BA")
plot!(ylabel = "Tamaño de la componente conexa más grande")
xlabel!("k")
end

# ╔═╡ 26bdc984-fabd-11eb-3d16-b9a85623c7fb
md"""Tal como vemos existe una relación inversa en el caso de ``n_0`` y una relación directa en el caso de ``k``.

Si bien, esto es una pequeña prueba que nos da una idea de lo que sucede con estos cambios de fase. Existen modelos probabilísticos, nada triviales, para determinar  las transiciones de fase de percolación para gráficas de libre escala como la de Barabasi-Albert que aquí presentamos."""

# ╔═╡ 1ab8c60e-fa1b-11eb-0c7e-b34fbfa7ecbb
md"### Conceptos básicos de percolación"

# ╔═╡ a8d72f88-fa1c-11eb-2a48-cb2ef72523c5
md"""
Generamos primero una retícula de tamaño ``L \times L`` de puntos que son ocupados con una probabilidad ``p``. Esta retícula corresponde con un medio de porosidad ``p``, consideramos que los sitios ocupados son agujeros en este material.
"""

# ╔═╡ cd7ab7a6-fa1e-11eb-32aa-7f63f6cac169
L = 50;

# ╔═╡ c4e27804-f62a-11eb-0d23-bf1696bb918e
init_random = rand(L,L);

# ╔═╡ d5dd16f0-f62a-11eb-0a97-4b049257f1e4
lattice_n = zeros(L,L);

# ╔═╡ 9b926992-f633-11eb-2eeb-33efe059c235
lattice_p = zeros(L,L);

# ╔═╡ d39d12ac-f637-11eb-2f46-c7acc5fd2020
for t in eachindex(init_random)
          if init_random[t] >=  0.75
                  lattice_n[t] = 1
		  else
			      lattice_n[t] = 0
       	  end
    end

# ╔═╡ 19a87a0a-f635-11eb-1bae-85b7a828b9b9
for t in eachindex(init_random)
          if init_random[t] >=  0.25
                  lattice_p[t] = 1
		  else
			      lattice_p[t] = 0
       	  end
    end

# ╔═╡ 468d78ee-f636-11eb-1005-f5b2b2bf30fa
heatmap(lattice_n)

# ╔═╡ 38e15dd6-f637-11eb-13c4-b7a7e0dbed27
heatmap(lattice_p)

# ╔═╡ 392b23ba-fa1c-11eb-227d-d12c53abb628
md""" Tal como vemos para probabilidades más altas, el material es más poroso. Mientras que para probabilidades menores el material es más sólido.
"""

# ╔═╡ 18e167ce-fa1f-11eb-1f7b-b98bd1ac47ae
md"""
Decimos que dos sitios están conectados si son los vecinos más cercanos. En el caso de una retícula cuadrada, tenemos 4.
"""

# ╔═╡ 5d80b556-fa1f-11eb-3cf1-6703efcbb300
md"""### Descripción del problema"""

# ╔═╡ 6d4cf788-fa1f-11eb-189a-b9a03686ba44
md""" Cuándo decimos que un sistema se percola? Pues, cuando existe una trayectoria que conecta a un extremo con otro.  Esto ocurre en algún valor ``p=p_c``. Sin embarho, en sistemas finitos, como el que estamos simulando, este valor ``p_c`` va variando en cada ejecución. Para caracterizar este comportamiento, introducimos el siguiente concepto:

``Π(p,L)``, la ``\textbf{probabilidad de percolación}``, es el valor para el cual existe una trayectoria que conecta un lado con otro como función de ``p`` en un sistema de tamaño ``L \times L.``

Para medir ``Π(p,L)``, en una muestra finita de tamaño ``L \times L``, generamos varias matrices aleatorias. Tal como lo hicimos en los ejemplos anteriores. Para cada matriz generada, realizamos un análisis de cluster para una secuencia de ``p_i`` valores. Para cada ``p_i`` encontramos todos los clusters(un cluster es un conjunto de sitios conexos). 

Si encontramos que alguno de estos clusters están presentes en ambos lados, derecho e izquierdo, sabremos que el sistema se percola. Luego, contamos cuantas veces un sistema se percola para una probabilidad ``p_i``, y el experimento ``N_i`` y luego dividimos por el numero total de experimentos ``N`` para estimar la probabilidad de percolación para una ``p_i``dado. Así, tenemos que:

$Π(pi,L) ≃ Ni/N$

"""

# ╔═╡ f2898b22-fa48-11eb-25fa-cdc3ad171372
md"""Generemos ahora los clusters para algunas probabilidades"""

# ╔═╡ 9e28637a-fa41-11eb-1d55-134644a4acb4
lwp, num1 = SciPy.ndimage.measurements.label(lattice_p)

# ╔═╡ aa2a7a06-fa47-11eb-2dc4-f958cbcc99ee
lwn, num2 = SciPy.ndimage.measurements.label(lattice_n)

# ╔═╡ c62a29f0-fa4e-11eb-15a4-81e1cab9bc54
md"Acá los clusters para la probabilidad ``p = 0.75``"

# ╔═╡ 0224742a-fa46-11eb-2509-c71f7cdf8dde
heatmap(lwn,c = :lightrainbow)

# ╔═╡ eedd8c00-fabe-11eb-1671-f3e031293385
md"""### Fuentes y trabajo futuro"""

# ╔═╡ 03dd6a52-fa48-11eb-1008-a58eae695216
md"""Falta definir una función que nos diga si existe una trayectoria perteneciente al mismo cluster."""

# ╔═╡ 5bdc56b2-fa4a-11eb-39cb-c50441e356cb
import Pkg;

# ╔═╡ dcd9fe36-fabd-11eb-32bb-eff3b7db9bf9
md"""
Fuentes:

Albert-Laszlo Barabasi. Network Science. http://networksciencebook.com/chapter/5#introduction5

Stauffer, D., & Aharony, A. (2018). Introduction to percolation theory. CRC press.
"""

# ╔═╡ Cell order:
# ╟─8e6d427c-f62a-11eb-131e-3357d8fad432
# ╟─91ebd6c0-fa4e-11eb-1453-e3217e276dfb
# ╠═ba216328-fa4c-11eb-0c08-fbbe22b707ab
# ╠═c2344572-fabe-11eb-1049-651fc12fe17b
# ╠═707c6828-fa4a-11eb-31a6-19b6713fd616
# ╟─e1c28daa-fab3-11eb-187d-3b4cb5788abb
# ╟─5ceaba4c-fab5-11eb-1935-4518850730f6
# ╠═5611c6cc-fa4f-11eb-3521-0f0d2fd67f72
# ╠═d355d844-fa4f-11eb-2f3b-496ab03ce67b
# ╠═d4783d24-fabe-11eb-1a65-a96571209d30
# ╟─e2afe1a6-fab6-11eb-08df-9baeedf9317a
# ╠═7b6b265e-fab6-11eb-3993-a1ea0ebe7a16
# ╟─176be4cc-fabb-11eb-05a4-db6723be5ed3
# ╠═22924a36-faba-11eb-2b50-c74fcf87051d
# ╠═7f3f3dd4-faba-11eb-3b10-bdf9961ffda1
# ╟─855e718e-fabb-11eb-050b-718ddc7256e7
# ╠═108fa28a-fab9-11eb-0ab8-1b971c2c0dbf
# ╠═20e16c34-fab9-11eb-1fd2-7599e4e4e1c9
# ╠═5eedfc0a-fab7-11eb-11e1-77aec110c445
# ╠═4f80321c-fab8-11eb-2581-8989e71dd9a2
# ╠═2f7ffae4-fab8-11eb-2e7b-11dc13b26851
# ╟─5bc5c400-fab9-11eb-114d-312f9139ae71
# ╠═6ade9dee-fabc-11eb-07c6-8fde2683c9e7
# ╠═48a0deac-fabc-11eb-242e-eb0df0411472
# ╠═443490b6-fabc-11eb-3028-956a647500a7
# ╟─a36620f4-fabc-11eb-1fbd-59a5d323dada
# ╟─26bdc984-fabd-11eb-3d16-b9a85623c7fb
# ╟─1ab8c60e-fa1b-11eb-0c7e-b34fbfa7ecbb
# ╟─a8d72f88-fa1c-11eb-2a48-cb2ef72523c5
# ╠═cd7ab7a6-fa1e-11eb-32aa-7f63f6cac169
# ╠═c4e27804-f62a-11eb-0d23-bf1696bb918e
# ╠═d5dd16f0-f62a-11eb-0a97-4b049257f1e4
# ╠═9b926992-f633-11eb-2eeb-33efe059c235
# ╠═d39d12ac-f637-11eb-2f46-c7acc5fd2020
# ╠═19a87a0a-f635-11eb-1bae-85b7a828b9b9
# ╠═df1964f2-f635-11eb-22b9-d987e3d2e5be
# ╠═468d78ee-f636-11eb-1005-f5b2b2bf30fa
# ╠═38e15dd6-f637-11eb-13c4-b7a7e0dbed27
# ╟─392b23ba-fa1c-11eb-227d-d12c53abb628
# ╟─18e167ce-fa1f-11eb-1f7b-b98bd1ac47ae
# ╟─5d80b556-fa1f-11eb-3cf1-6703efcbb300
# ╟─6d4cf788-fa1f-11eb-189a-b9a03686ba44
# ╠═1491f03a-fa47-11eb-023d-5fea359ab4a8
# ╟─f2898b22-fa48-11eb-25fa-cdc3ad171372
# ╠═9e28637a-fa41-11eb-1d55-134644a4acb4
# ╠═aa2a7a06-fa47-11eb-2dc4-f958cbcc99ee
# ╟─c62a29f0-fa4e-11eb-15a4-81e1cab9bc54
# ╠═0224742a-fa46-11eb-2509-c71f7cdf8dde
# ╟─eedd8c00-fabe-11eb-1671-f3e031293385
# ╟─03dd6a52-fa48-11eb-1008-a58eae695216
# ╠═5bdc56b2-fa4a-11eb-39cb-c50441e356cb
# ╠═8dc882cc-fa4f-11eb-23d9-1d852138ab04
# ╟─dcd9fe36-fabd-11eb-32bb-eff3b7db9bf9
