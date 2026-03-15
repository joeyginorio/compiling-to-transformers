# Compiling to transformers

The following is a formal specification of the programming language $\cj$. It includes definitions of its syntax, its operational semantics, its typing relation, and its compiler. These specifications are checked by proving both that programs produce behavior (well-typed programs terminate) and that such programs compile correctly to linear maps (compiler preserves program behavior). 

## Syntax

$$\nb{\ty} \in \nb{\t{Type}}$$
$$\begin{aligned} 
 \nb{\tau} \,\,\,\coloneqq 
	 &\ \ \ \nb{\Unit} &&\t{unit}\\
	 &\mid \nb{\ty_1 \x \ty_2} &&\t{product}\\
	 &\mid \nb{\ty_1 \+ \ty_2} &&\t{sum}\\
	 &\mid \nb{\ty_1 \rightharpoonup \ty_2} &&\t{dictionary}\\
 \end{aligned}$$

$$\nb{e} \in \nb{\t{Expression}}$$
$$\begin{aligned} 
 \nb{e} \,\,\,\coloneqq 
	 \ & \ \ \ \nb{x} &&\t{variable}\\
	 &\mid \unit &&\t{unit}\\
     &\mid \nb{\t{error}} &&\t{error}\\
	 &\mid \nb{(e_1,e_2)} &&\t{tuple}\\
	 &\mid \nb{\inj{i}{e}} &&\t{injection}\\
	 &\mid \nb{\dict{e_1}{e_2}} &&\t{dictionary}\\
	 &\mid \nb{\proj{i}{e}} &&\t{projection}\\
	 &\mid \nb{\case{e}{x_1}{e_1}{x_2}{e_2}} &&\t{case analysis}\\
	 &\mid \nb{e_1 \sqcap e_2} &&\t{nondeterministic choice}\\
	 &\mid \nb{e_1(e_2)_\mathcal{R}} &&\t{lookup}\\
	 &\mid \nb{\let{x}{e_1}{e_2}} &&\t{sequential composition}\\
\end{aligned}$$

$$\nb{v} \in \nb{\textsf{Value}}$$
$$\begin{aligned}
 \nb{v} \,\,\,\coloneqq 
	 &\ \ \ \unit && \t{unit}\\
	 &\mid \nb{\inj{i}{v}} && \t{injection}\\
	 &\mid \nb{(e_1,e_2)} && \t{tuple}\\
	 &\mid \nb{\dict{v_1}{v_2}} &&\t{dictionary}\\
	 &\mid \nb{\t{error}} &&\t{error}\\

 \end{aligned}$$

$$\nb{\D} \in \nb{\t{Context}}$$
$$\begin{aligned} 
 \nb{\D} \coloneqq \nb{\varnothing} \mid \nb{\D,\bind{x}{\tau}}
 \end{aligned}$$

## Dynamics
$$\nb{\_} \bstep \nb{\_} \subseteq \nb{\textsf{Expression}} \times \nb{\textsf{Value}}$$

$$
\begin{prooftree}
\AxiomC{}
\UnaryInfC{$\step{v}{v}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\step{e}{v}$}
\UnaryInfC{$\step{\inj{i}{e}}{\inj{i}{v}}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\step{e}{(e_1,e_2)}$}
\AxiomC{$\step{e_1}{v_1}$}
\BinaryInfC{$\step{\proj{1}{e}}{v_1}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\step{e}{(e_1,e_2)}$}
\AxiomC{$\step{e_2}{v_2}$}
\BinaryInfC{$\step{\proj{2}{e}}{v_2}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\step{e}{\t{error}}$}
\UnaryInfC{$\step{E[e]}{\t{error}}$}
\end{prooftree}
$$
$$
\begin{prooftree}
  \AxiomC{$\step {e_1} {v_1}$}
  \AxiomC{$\step {\{x \map v_1\}(e_2)} {v_2}$}
  \BinaryInfC{$\step {\let{x}{e_1}{e_2}} {v_2}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\ess1 \bstep \vss1$}
\UnaryInfC{$\ess1 \sqcap \ess2 \bstep \vss1$} 
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\ess2 \bstep \vss2$}
\UnaryInfC{$\ess1 \sqcap \ess2 \bstep \vss2$} 
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\step{e_\t{k}}{v_\t{k}}$}
\AxiomC{$\step{e_\t{v}}{v_\t{v}}$}
\BinaryInfC{$\step{\dict{e_\t{k}}{e_\t{v}}}{\dict{v_\t{k}}{v_\t{v}}}$}
\end{prooftree}
$$
$$
\begin{prooftree}
\AxiomC{$\step{e}{\inj{1}{v}}$}
\AxiomC{$\step{\{x_1 \map v\}(e_1)}{v_1}$}
\BinaryInfC{$\step{\case{e}{x_1}{e_1}{x_2}{e_2}}{v_1}$}
\end{prooftree}
\quad
\quad
\begin{prooftree}
\AxiomC{$\step{e}{\inj{2}{v}}$}
\AxiomC{$\step{\{x_2 \map v\}(e_2)}{v_2}$}
\BinaryInfC{$\step{\case{e}{x_1}{e_1}{x_2}{e_2}}{v_2}$}
\end{prooftree}
$$
$$
\begin{prooftree}
\AxiomC{$\step{e_\t{kv}}{\dict{v_{\t{k}}}{v_\t{v}}}$}
\AxiomC{$\step{e_\t{q}}{v_\t{q}}$}
\AxiomC{$\nb{\mathcal{V}} = \{\nb{(v_\t{v})_i} \mid (\nb{v_\t{q}},\nb{(v_\t{k})_i}) \in \nb{\mathcal{R}}\}$}
\AxiomC{${\Large \sqcap}_{\nb{\mathcal{V}}}\,\nb{v}\bstep \vss{\t{r}}$}
\QuaternaryInfC{$\step{e_\t{kv}(e_\t{q})_\mathcal{R}}{v_\t{r}}$}
\end{prooftree}
$$


- TODO: Define evaluation contexts.
- TODO: Define ${\large \sqcap}_\mathcal{V}$ by cases, if it's empty return error. Otherwise nondeterministically choose from the set of values.

## Typing

$$\nb{\_} \- \nb{\_} : \nb{\_} \subseteq \nb{\textsf{Context}} \times \nb{\textsf{Expression}} \times \nb{\textsf{Type}}$$
$$
\begin{prooftree}
\AxiomC{}
\UnaryInfC{$\p{\bind{x}{\ty}} {x} {\ty}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{}
\UnaryInfC{$\p{\emp} {\unit} {\Unit}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{}
\UnaryInfC{$\p{\D} {\t{error}} {\ty}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\p{\D}{e}{\ty_i}$}
\UnaryInfC{$\p{\D} {\inj{i}{e}} {\ty_1 \+ \ty_2}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\p{\D}{e_1}{\ty_1}$}
\AxiomC{$\p{\D}{e_2}{\ty_2}$}
\BinaryInfC{$\p{\D} {(e_1,e_2)} {\ty_1 \x \ty_2}$}
\end{prooftree}
$$
$$
\begin{prooftree}
\AxiomC{$\p{\D_\t{k}}{e_\t{k}}{\t{List}_n(\nb{\t{enum}}(\nb{\ty_\t{k}}))}$}
\AxiomC{$\p{\D_\t{v}}{e_\t{v}}{\t{List}_n(\ty_\t{v})}$}
\BinaryInfC{$\p{\D_\t{k} \o \D_\t{v}} {\dict{e_\t{k}}{e_\t{v}}} {\ty_\t{k} \Map \ty_\t{v}}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\p{\D_1}{e_1}{\ty_1}$}
\AxiomC{$\p{\D_2,\bind{x}{\ty_1}}{e_2}{\ty_2}$}
\BinaryInfC{$\p{\D_1 \o \D_2} {\let{x}{e_1}{e_2}} {\ty_2}$}
\end{prooftree}
$$
$$
\begin{prooftree}
\AxiomC{$\p{\D_1}{e}{\ty_1 \+ \ty_2}$}
\AxiomC{$\p{\D_2,\bind{x_1}{\ty_1}}{e_1}{\ty}$}
\AxiomC{$\p{\D_2,\bind{x_2}{\ty_2}}{e_2}{\ty}$}
\TrinaryInfC{$\p{\D_1 \o \D_2} {\case{e}{x_1}{e_1}{x_2}{e_2}} {\ty}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\p{\D}{e}{\ty_1 \x \ty_2}$}
\UnaryInfC{$\p{\D} {\pi_ie} {\ty_i}$}
\end{prooftree}
$$
$$
\begin{prooftree}
\AxiomC{$\p{\D_\t{kv}}{e_\t{kv}}{\ty_\t{k} \Map \ty_\t{v}}$}
\AxiomC{$\p{\D_\t{q}}{e_\t{q}}{\nb{\t{enum}}(\nb{\ty_\t{q}})}$}
\AxiomC{$\nb{\mathcal{R}}\subseteq \nb{\t{Val}}(\nb{\ty_\t{q}}) \x \nb{\t{Val}}(\nb{\ty_\t{k}})$}
\TrinaryInfC{$\p{\D_\t{kv} \o \D_\t{q}} {e_\t{kv}(e_\t{q})_\mathcal{R}} {\ty_\t{v}}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\p{\D}{e_1}{\ty}$}
\AxiomC{$\p{\D}{e_2}{\ty}$}
\BinaryInfC{$\p{\D} {e_1 \sqcap e_2} {\ty}$}
\end{prooftree}
$$

TODO: Define $\t{enum}$ relation on types.

###### Definition (Source environment)
$$
\begin{prooftree}
\AxiomC{}
\UnaryInfC{$\nb{\emp} \vdash \nb{\emp}$}
\end{prooftree}
\quad\quad
\begin{prooftree}
\AxiomC{$\nb{\D}\-\nb{\nv}$}
\AxiomC{$\p{\emp}{v}{\ty} $}
\BinaryInfC{$\nb{\D,\bind{x}{\ty}} \vdash \nb{\nv\{x \map v\}}$}
\end{prooftree}
$$

## Compiling

###### Definition (Compiling types)
$$\c{\nb{\_}}: \nb{\t{Type}} \to \rd{\t{Inner Product Space}}$$
$$
\begin{aligned}
\cm{\Unit} &= \rd{\R}\\
\c{\nb{\ty_1 \+ \ty_2}} &= \c{\nb{\ty_1}} \,\rd{\+}\,\c{\nb{\ty_2}}\\
\c{\nb{\ty_1 \x \ty_2}} &= \c{\nb{\ty_1}} \,\rd{\x}\,\c{\nb{\ty_2}}\\
\c{\nb{\ty_\t{k} \Map \ty_\t{v}}} &= \rd{\t{Lin}}(\c{\nb{\ty_\t{k}}} ,\c{\nb{\ty_\t{v}}})\\
\end{aligned}
$$
TODO: Define the inner product in each case. Choice is not unique, so it is important to specify. Use dot product on $\R^n$, Frobenius inner product on function spaces.

###### Definition (Compiling contexts)
$$\c{\nb{\_}}: \nb{\t{Context}}   \to \rd{\t{Tuple}}(\rd{\t{Vector}})$$ 
$$
\begin{aligned}
\c{\nb{\emp}} &= \rd{\{0\}}\\
\c{\nb{\D,\bind{x}{\ty}}} &= \c{\nb{\D}}\,\rd{\x}\,\c{\nb{\ty}}\\ 
\end{aligned} 
$$

###### Definition (Compiling programs)

Note: By convention, the environment is supplied to subprograms is restricted according to the context specified in its typing judgment.

$$\imp{\p{\D} {e} {\ty}}{\svec}={\et{}} \iff \t{program }\nb{e} \t{ is implemented by vector }\rd{\et{}}\t{ under environment }\rd{\sig}$$
$$\cm{-}:\nb{\t{Program}}\to \rd{\t{Multilinear map}}$$
$$
\begin{aligned}
&\c{\p{\bind{x}{\ty}}{x}{\ty}}({\xvec}) = \xvec\\\\
&\c{\p{\emp}{\unit}{\Unit}}(\rd{0}) = \rd{1}\\\\
&\c{\p{\D}{\t{error}}{\ty}}(\svec) =\rd{0}_{[\hspace{-.18em}[\nb{\ty}]\hspace{-.18em}]}\\\\
&\c{\p{\D}{\inj{1}{e}}{\ty_1 \+ \ty_2}}(\svec) = \rd{(}\cm{e}\rd{,}\,\rd{0}_{[\hspace{-.18em}[\nb{\ty_2}]\hspace{-.18em}]}\rd{)}\\\\
&\c{\p{\D}{\inj{2}{e}}{\ty_1 \+ \ty_2}}(\svec) = \rd{(}\rd{0}_{[\hspace{-.18em}[\nb{\ty_2}]\hspace{-.18em}]}\rd{,}\,\cm{e}\rd{)}\\\\
&\c{\p{\D}{(e_1,e_2)}{\ty_1 \x \ty_2}}(\svec) = \rd{(}\cm{e_1}\rd{,}\,\cm{e_2}\rd{)}\\\\
&\c{\p{\D_\t{k} \o \D_\t{v}}{\dict{e_\t{k}}{e_\t{v}}}{\ty_\t{k} \Map \ty_\t{v}}}(\svec) = \xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\,\rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\ess{\t{k}}}_\rd{i}\rd{,\,\xvec_\t{q}\rangle} \rt \cm{\ess{\t{v}}}_\rd{i}\\\\
&\c{\p{\D_1 \o \D_2}{\let{x}{e_1}{e_2}}{\ty_2}}(\svec) = \cm{e_2}(\cm{e_1})\\\\
&\c{\p{\D_1 \o \D_2}{\case{e}{x_1}{e_1}{x_2}{e_2}}{\ty_2}}(\svec) = \cm{e_1}(\cm{e}_\rd{1}) \rp \cm{e_2}(\cm{e}_\rd{2})\\\\
&\c{\p{\D}{\pi_ie}{\ty_i}}(\svec) = \cm{e}_\rd{i}\\\\
&\c{\p{\D_\t{kv}\o\D_\t{q}}{e_\t{kv}(e_\t{q})_\mathcal{R}}{\ty_\t{v}}}(\svec) = \cm{e_\t{kv}}(\cm{\mathcal{R}}(\cm{e_\t{q}}))\\\\
&\c{\p{\D}{e_1 \sqcap e_2}{\ty}}(\svec) = \cm{e_1}\rp\cm{e_2}\\\\
\end{aligned}
$$
## Metatheory

Assume Barendregt's convention.

TODO: Define capture-avoiding substitution. Informally is fine.
###### Theorem (Programs have behavior)
If $\p{\D}{e}{\ty}$ and $\D\vdash\nv$, then
$$\exists \vs, \nv(\es) \bstep \vs$$
**Proof.** By induction on $\p{\D}{e}{\ty}$,
- **Case** $\p{\emp}{\unit}{\Unit}$
	Let $\exists \vs = \unit$
	By evaluation $\unit \bstep \unit$
- **Case** $\p{\D}{\error}{\ty}$
	Let $\exists \vs = \error$
	By substitution and evaluation $\nv(\error)=\error \bstep \error$ 
- **Case** $\p{\bind{x}{\ty}}{x}{\ty}$
	By assumption $\nb{\bind{x}{\ty}}\vdash \nb{\{x \map v\}}$ where $\vs :\nb{\ty}$
	Let $\exists \vs= \nv(\xs)$
	By substitution and evaluation $\nv(\xs)=\vs\bstep \vs$
- **Case** $\p{\D}{\inj{1}{e}}{\ty_1 \+ \ty_2}$ 
	By induction $\nv(\es) \bstep \vss1$
	Let $\exists \vs=\inj{1}{v_1}$ 
	By substitution and evaluation $\nv(\inl{e})=\inj{1}{\nv(e)} \bstep \inj{1}{v_1}$ 
- **Case** $\p{\D}{\inj{2}{e}}{\ty_1 \+ \ty_2}$ 
	Similar to previous case
- **Case** $\p{\D}{(e_1,e_2)}{\ty_1 \x \ty_2}$
	Let $\exists \vs=\nb{(\nv(\ess1),\nv\nb(\ess2))}$
	By substitution and evaluation $\nv(\nb{(\ess1,\ess2)})=\nb{(\nv(e_1),\nv(e_2))} \bstep \nb{(\nv(e_1),\nv(e_2))}$ 
- **Case** $\p{\D_1 \o \D_2}{\dict{e_1}{e_2}}{\ty_1 \Map \ty_2}$ 
	By splitting source environments $\nv=\nvs1 \o \nvs2$ where $\Ds1 \vdash \nvs1$ and $\Ds2 \vdash \nvs2$
	By induction and substitution $\nvs1(\ess1) = \nv(\ess1)\bstep \vss1$
	By induction and substitution $\nvs2(\ess2) = \nv(\ess2)\bstep \vss2$
	Let $\exists \vs=\dict{\vss1}{\vss2}$
	By substitution and evaluation $\nv(\dict{e_1}{e_2})=\dict{\nv(e_1)}{\nv(e_2)}=\dict{\nvs1(e_1)}{\nvs2(e_2)} \bstep \dict{v_1}{e_2}$
- **Case** $\p{\D_1 \o \D_2}{\let{x}{e_1}{e_2}}{\ty_2}$
	By splitting source environments $\nv=\nvs1 \o \nvs2$ where $\Ds1 \vdash \nvs1$ and $\Ds2 \vdash \nvs2$
	By induction and substitution $\nvs1(\ess1) = \nv(\ess1) \bstep \vss1$ 
	By source environments $\nb{\Ds2,\bind{x}{\ty_1}} \- \nb{\nvs2\{x \map v_1\}}$
	By induction and substitution $\nb{\nvs2\{x \map v_1\}}(\ess2)=  \nb{\nv\{x \map v_1\}}(\ess2)\bstep \vss2$ 
	Because substitutions commute $\nb{\{x \map v_1\}}(\nb{\nv}(\ess2))\bstep \vss2$
	Let $\exists \vs = \vss2$
	By substitution and evaluation $\nv(\let{x}{e_1}{e_2})=(\let{x}{\nvs1(e_1)}{\nvs2(e_2)}) \bstep \vss2$ 
- **Case** $\p{\D_1 \o \D_2}{\sume{e}{x_1}{e_1}{x_2}{e_2}}{\ty}$
	By splitting source environments $\nv=\nvs1 \o \nvs2$ where $\Ds1 \vdash \nvs1$ and $\Ds2 \vdash \nvs2$
	By induction and substitution $\nvs1(\es) = \nv(\es) \bstep \vs$
	Because evaluation preserves typing and by closing substitutions $\vs:\nb{\ty_1 \+ \ty_2}$ 
	By canonical forms either $\vs=\inl{v_1}$ or $\vs=\inr{v_2}$
	Consider each possibility,
	- **Case** $\vs=\inl{v_1}$
		By source environments $\nb{\Ds2,\bind{x}{\ty_1}} \- \nb{\nvs2\{x \map v_1\}}$
		By induction and substitution $\nb{\nvs2\{x \map v_1\}}(\ess1)=\nb{\nv\{x \map v_1\}}(\ess1) \bstep \vss{\bullet}$ 
		Because substitutions commute $\nb{\{x \map v_1\}}(\nb{\nv}(\ess1)) \bstep \vss{\bullet}$
		Let $\exists \vs=\vss\bullet$
		By substitution and evaluation $\nb{\sume{\nv(e)}{x_1}{\nv(e_1)}{x_2}{\nv(e_2)}} \bstep \vss\bullet$
	- **Case** $\vs = \inr{v_2}$
		Similar to previous case
- **Case** $\p{\D}{\pi_1e}{\ty_1}$
	By induction $\nv(\es) \bstep \vs$ 
	Because evaluation preserves typing and by closing substitutions $\vs:\nb{\ty_1 \x \ty_2}$
	By canonical forms $\vs=\nb{(v_1,v_2)}$
	Let $\exists \vs=\vss1$
	By substitution and evaluation $\nv(\nb{\pi_1e})=\nb{\pi_1\nv(e)} \bstep \vss1$  
- **Case** $\p{\D}{\pi_2e}{\ty_2}$
	Similar to previous case
- **Case** $\p{\D}{e_1 \sqcap e_2}{\ty}$
	By induction $\nv(\ess1) \bstep \vss1$
	By induction $\nv(\ess2) \bstep \vss2$
	Let $\exists \vs=\vss1$
	By substitution and evaluation $\nv(\nb{e_1 \sqcap e_2})=\nb{\nv(\ess1) \sqcap \nv(\ess2)} \bstep \vss1$ 
- **Case** $\p{\D_\t{kv} \o \D_\t{q}}{e_\t{kv}(e_\t{q})_\mathcal{R}}{\ty_\t{v}}$
	By splitting source environments $\nv=\nvs{\t{kv}} \o \nvs{\t{q}}$ where $\Ds{\t{kv}} \vdash \nvs{\t{kv}}$ and $\Ds{\t{q}} \vdash \nvs{\t{q}}$ 
	By induction and substitution $\nvs{\t{kv}}(\ess{\t{kv}})=\nv(\ess{\t{kv}}) \bstep \vss{\t{kv}}$ 
	By induction and substitution $\nvs{\t{q}}(\ess{\t{q}})=\nv(\ess{\t{q}}) \bstep \vss{\t{q}}$ 
	Let $\nb{\mathcal{V}} = \{\nb{(v_\t{v})_i} \mid (\nb{e_\t{q}},\nb{(e_\t{k})_i}) \in \nb{\mathcal{R}}\}$
	Consider whether $\mathcal{\nb{V}}$ is empty,
	- **Case** $\nb{\mathcal{V}}=\nb{\emp}$ 
		By nondeterministic summation ${\Large \sqcap}_\nb{\emp}=\nb{\t{error}}$
		Let $\exists \vs=\nb{\t{error}}$
		By evaluation $\nb{\t{error}} \bstep \nb{\t{error}}$ 
		By substitution and evaluation $\nv(\ess{\t{kv}}(\ess{\t{q}})_{\nb{\mathcal{R}}})=\nvs{\t{kv}}(\ess{\t{kv}})(\nvs{\t{q}}(\ess{\t{q}}))_{\nb{\mathcal{R}}} \bstep \nb{\t{error}}$ 
	- **Case** $\nb{\mathcal{V}} \neq \nb{\emp}$
		Let $\vss{\t{r}} \in \nb{\mathcal{V}}$
		Let $\exists \vs = \vss{\t{r}}$
		Because nondeterministic sums of values return any summand ${\Large \sqcap}_\nb{\mathcal{V}}\,\vs \bstep \vss{\t{r}}$
		By substitution and evaluation $\nv(\ess{\t{kv}}(\ess{\t{q}})_{\nb{\mathcal{R}}})=\nv(\ess{\t{kv}})(\nv(\ess{\t{q}}))_{\nb{\mathcal{R}}} \bstep \vss{\t{r}}$  

$$\blacksquare$$


###### Definition (Derivation of $\es\bstep\vs$)
The derivation $\der$ associated with $\es\bstep \vs$ represents *the execution trace*.
$$\es \bstep_\nb{D}\vs \iff \nb{D} \text{ is a derivation of }\es \bstep \vs$$
###### Theorem (Compiler preserves program behavior)
If $\p{\D}{e}{\ty}$ and $\D \vdash \nv$ and $\nb{V}=\{(\nb{D},\vs) \mid \nv(\es) \bstep_\nb{D} \vs\}$,
$$\cm{e}(\cm{\nv})=\rd{\sum_{\bl{(\nb{D}\hspace{.05em},\hspace{.05em}\vs)\in\nb{V}}}}\,\cm{v}$$

**Proof.** By induction on $\p{\D}{e}{\ty}$
- **Case** $\p{\bind{x}{\ty}}{x}{\ty}$
	**Show** $\cm{x}(\cm{v})=\rsum_{(\der,\,\vs)\in\Vs}\cm{v}$
	By assumption $\nb{\bind{x}{\ty}}\vdash\nb{\{x \map v\}}$ 
	By assumption $\Vs=\{(\der,\vs) \mid \nb{\{x \mapsto v\}(x)} \bstep \vs\}$ 
$$
\begin{aligned}
                &\,\,\;\;\;\, \cm{x}(\cm{v}) \\[.2em]
		        &= \cm{v} && \t{By compiling}\\[.2em]
		        &= \rsum_{(\der,\,\vs)\in\Vs}\cm{v} && \t{Because }\Vs=\{(\der,\vs)\}\\
\end{aligned}
$$
- **Case** $\p{\emp}{\unit}{\Unit}$
	**Show** $\cm{\unit}=\rsum_{(\der,\,\unit)\in\Vs}\cm{\unit}$ 
	By assumption $\Vs=\{(\der,\unit) \mid \unit \bstep \unit\}$
$$
\begin{aligned}
                &\,\,\;\;\;\, \cm{\unit} \\[.2em]
		        &= \rsum_{(\der,\,\unit)\in\Vs}\cm{\unit} && \t{Because }\Vs=\{(\der,\unit)\}
\end{aligned}
$$
- **Case** $\p{\D}{\t{error}}{\ty}$
	**Show** $\cm{\t{error}}(\cm{\nv})=\rsum_{(\der,\,\nb{\t{error}})\in\Vs}\cm{\t{error}}$ 
	By assumption $\Vs=\{(\der,\nb{\t{error}}) \mid \error \bstep \error\}$
$$
\begin{aligned}
                &\,\,\;\;\;\, \cm{\error}(\cm{\nv})\\[.2em]
		        &= \rsum_{(\der,\,\error)\in\Vs}\cm{\error} && \t{Because }\Vs=\{(\der,\error)\}
\end{aligned}
$$
- **Case** $\p{\D}{\inj{1}{e}}{\ty_1 \+ \ty_2}$
	**Show** $\cm{\inj{1}{e}}(\cm{\nv})=\rsum_{(\der,\,\vs)\in\Vs}\cm{\vs}$ 
	By assumption $\nb{V}=\{(\nb{D},\vs) \mid \nv(\inj{1}{e})=\inj{1}{\nv(e)}\bstep_\nb{D}\vs\}$ 
	Let $\nb{\bar{V}}=\{(\nb{\bar{D}},\nb{\bar{\vs}}) \mid \nv(\es) \bstep_\nb{\bar{D}}\nb{\bar{\vs}}\}$ by inversion on derivations in $\nb{V}$
	Observe that $(\nb{D},\vs) \in\Vs \iff (\nb{\bar{D}},\nb{\bar{v}})\in \nb{\bar{V}}$ when $\nb{\bar{D}}$ is obtained by inversion on $\nb{D}$

$$
\begin{aligned}
                &\,\,\;\;\;\, \cm{\inj{1}{e}}(\cm{\nv}) \\[.2em]
		        &= \rd{(}\cm{e}\rd{,0)} && \t{By compiling}\\[.2em]
		        &= \rd{(}\rd{\sum}_{(\nb{\bar{D}},\,\nb{\bar{v}})\in\nb{\bar{V}}}\cm{\bar{v}}\rd{,0)} && \t{By induction}\\[.2em]
		        &=\rd{\sum}_{(\nb{\bar{D}},\,\nb{\bar{v}})\in\nb{\bar{V}}} \rd{(}\cm{\bar{v}}\rd{,0)} && \t{Because }\rd{(a+b,0)}=\rd{(a,0)+(b,0)}\\[.2em]
		        &=\rd{\sum}_{(\nb{D},\,\vs)\in\nb{V}} \rd{(}\cm{v}\rd{,0)} && \t{Because }(\nb{D},\vs) \in\Vs \iff (\nb{\bar{D}},\nb{\bar{v}})\in \nb{\bar{V}}\\[.2em]
		        &=\rd{\sum}_{(\nb{D},\,\vs)\in\nb{V}} \cm{\inj{1}{v}} && \t{By compiling}\\ 
\end{aligned}
$$
- **Case** $\p{\D}{\inj{2}{e}}{\ty_1 \+ \ty_2}$
	Similar to previous case
- **Case** $\p{\D}{(e_1,e_2)}{\ty_1 \x \ty_2}$
	**Show** $\cm{(e_1,e_2)}(\cm{\nv})=\rsum_{(\der,\,\nb{\nv(e_1,e_2)})\in\Vs}\cm{\nv(e_1,e_2)}$ 
	By assumption $\nb{V}=\{(\nb{D},\nb{(e_1,e_2)}) \mid \nv(\nb{(e_1,e_2)})=\nb{(\nv(e_1),\nv(e_2))}\bstep_\nb{D}\nb{(\nv(e_1),\nv(e_2))}\}$ 
	Observe that $\Vs=\{(\der,\nb{(\nv(e_1),\nv(e_2))}\}=\{(\der,\nb{\nv(e_1,e_2)})\}$ 
$$
\begin{aligned}
                &\,\,\;\;\;\, \cm{(e_1,e_2)}(\cm{\nv}) \\[.2em]
		        &= \cm{\nv(e_1,e_2)} && \t{Because compiling commutes with substitution}\\[.2em]	        
                &= \rsum_{(\der,\,\nb{\nv(e_1,\,e_2)})\in\Vs}\cm{\nv(e_1,\,e_2)} && \t{Because }\Vs=\{(\der, \nb{\nv(e_1,e_2)})\}\\
\end{aligned}
$$
- **Case** $\p{\Ds{\t{k}}\o\Ds{\t{v}}}{\dict{e_\t{k}}{e_\t{v}}}{\ty_\t{k} \Map \ty_\t{v}}$
	**Show** $\cm{\dict{e_\t{k}}{e_\t{v}}}(\cm{\nv})=\rsum_{(\der,\,\dict{v_\t{k}}{v_\t{v}})\in\Vs}\cm{\dict{v_\t{k}}{v_\t{v}}}$ 
	By assumption $\nb{V}=\{(\nb{D},\dict{v_\t{k}}{v_\t{v}}) \mid \nv(\dict{e_\t{k}}{e_\t{v}})\bstep_\nb{D}\dict{v_\t{k}}{v_\t{v}}\}$
	Let $\nb{\Vss{\t{k}}}=\{(\nb{\ders{\t{k}}},\vss{\t{k}}) \mid \nv(\ess{\t{k}}) \bstep_\nb{\ders{\t{k}}}\vss{\t{k}}\}$ by inversion on derivations in $\nb{V}$
	Let $\nb{\Vss{\t{v}}}=\{(\nb{\ders{\t{v}}},\vss{\t{v}}) \mid \nv(\ess{\t{v}}) \bstep_\nb{\ders{\t{v}}}\vss{\t{v}}\}$ by inversion on derivations in $\nb{V}$
	Observe that $(\nb{D},\dict{v_\t{k}}{v_\t{v}}) \in\Vs \iff (\nb{\ders{\t{k}}},\vss{\t{k}})\in \Vss{\t{k}} \;\land\; (\nb{\ders{\t{v}}},\vss{\t{v}})\in \Vss{\t{v}}$ when $\nb{\ders{\t{k}}}$ and $\ders{\t{v}}$ are obtained by inversion on $\nb{D}$
	$$
\begin{aligned}
                &\,\,\;\;\;\, \cm{\dict{e_\t{k}}{e_\t{v}}}(\cm{\nv}) \\[.2em]
		        &=\Bigl(\xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\,\rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\ess{\t{k}}}_\rd{i}\rd{,\,\xvec_\t{q}\rangle} \rt \cm{\ess{\t{v}}}_\rd{i}\Bigr) && \t{By compiling}\\[.2em]
		        &= \Bigl(\xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\,\rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\rsum_{(\ders{\t{k}},\vss{\t{k}})\in\Vss{\t{k}}}\cm{\vss{\t{k}}}_\rd{i}\rd{,\,\xvec_\t{q}\rangle} \rt \Bigl(\rsum_{(\ders{\t{v}},\vss{\t{v}})\in\Vss{\t{v}}}\cm{\vss{\t{v}}}_\rd{i}\Bigr)\Bigr) && \t{By induction}\\[.2em]
		        &= \rsum_{(\ders{\t{k}},\vss{\t{k}})\in\Vss{\t{k}}}\rsum_{(\ders{\t{v}},\vss{\t{v}})\in\Vss{\t{v}}}\Bigl(\xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\,\rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\vss{\t{k}}}_\rd{i}\rd{,\,\xvec_\t{q}\rangle} \rt \cm{\vss{\t{v}}}_\rd{i}\Bigr) && \t{Because }\cm{\dict{e_\t{k}}{e_\t{v}}}\t{ is linear in } \cm{e_\t{k}} \t{ and }\cm{e_\t{v}}\\[.2em]
		        &=\rd{\sum}_{(\nb{D},\,\dict{v_\t{k}}{v_\t{v}})\in\nb{V}}  \Bigl(\xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\,\rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\vss{\t{k}}}_\rd{i}\rd{,\,\xvec_\t{q}\rangle} \rt \cm{\vss{\t{v}}}_\rd{i}\Bigr) && \t{Because }(\nb{D},\dict{v_\t{k}}{v_\t{v}}) \in\Vs \iff (\nb{\ders{\t{k}}},\vss{\t{k}})\in \Vss{\t{k}} \;\land\; (\nb{\ders{\t{v}}},\vss{\t{v}})\in \Vss{\t{v}}\\[.2em]
		        &=\rd{\sum}_{(\nb{D},\,\dict{v_\t{k}}{v_\t{v}})\in\nb{V}} \cm{\dict{v_\t{k}}{v_\t{v}}} && \t{By compiling}\\ 
\end{aligned}
$$

- **Case** $\p{\D_1 \o \D_2}{\let{x}{e_1}{e_2}}{\ty_2}$
	**Show** $\cm{\let{x}{e_1}{e_2}}(\cm{\nv})=\rsum_{(\der,\,\vss2)\in\Vs}\cm{\vss2}$ 
	By assumption $\nb{V}=\{(\nb{D},\vss2) \mid \nv(\let{x}{e_1}{e_2})\bstep_\nb{D}\vss2\}$
	Let $\nb{\Vss{1}}=\{(\nb{\ders{1}},\vss{1}) \mid \nv(\ess{1}) \bstep_\nb{\ders{1}}\vss{1}\}$ by inversion on derivations in $\nb{V}$
	Let $\nb{\Vss{2}}=\{(\nb{\ders{2}},\vss{2})_{} \mid (\_,\vss1)\in\Vss1,\,\nb{\{x \map v_1\}}(\nv(\ess{2})) \bstep_\nb{\ders{2}}\vss{2}\}$ by inversion on derivations in $\nb{V}$
	Observe that $(\nb{D},\vss2) \in \Vs \iff (\nb{\ders{1}},\vss{1})\in \Vss{1} \;\land\; (\nb{\ders{2}},\vss{2})\in \Vss{2}$ when $\nb{\ders{1}}$ and $\ders{2}$ are obtained by inversion on $\nb{D}$
	$$
\begin{aligned}
                &\,\,\;\;\;\, \cm{\let{x}{e_1}{e_2}}(\cm{\nv}) \\[.2em]
		        &=\cm{e_2}(\cm{e_1}) && \t{By compiling}\\[.2em]
		        &=\cm{e_2}\Bigl(\rsum_{(\ders1,\,\vss1)\in\Vss1}\cm{v_1}\Bigr) && \t{By induction}\\[.2em]		        &=\rsum_{(\ders1,\,\vss1)\in\Vss1}\cm{e_2}(\cm{v_1}) && \t{Because }\cm{e_2} \t{ is linear}\\[.2em]
                &=\rsum_{(\ders1,\,\vss1)\in\Vss1}\cm{\{x \map v_1\}(e_2)} && \t{Because compiling commutes with substitution}\\    &=\rsum_{(\ders1,\,\vss1)\in\Vss1}\rsum_{(\ders2,\,\vss2)\in\Vss2}\cm{v_2} && \t{By induction}\\[.2em]
                &=\rsum_{(\der,\,\vss2)\in\Vs}\cm{v_2} && \t{Because }(\nb{D},\vss2) \in \Vs \iff (\nb{\ders{1}},\vss{1})\in \Vss{1} \;\land\; (\nb{\ders{2}},\vss{2})\in \Vss{2}\\[.2em]
\end{aligned}
$$
- **Case** $\p{\D_1\o\D_2}{\case{e}{x_1}{e_1}{x_2}{e_2}}{\ty_2}$
	By assumption $\Vs=\{(\der,\vs) \mid \nv(\nb{\case{e}{x_1}{e_1}{x_2}{e_2}}) \bstep \vs\}$
	Let $\nb{\Vss{12}}=\{(\nb{\ders{12}},\vss{12}) \mid \nv(\es) \bstep_\nb{\ders{12}}\vss{12}\}$ by inversion on derivations in $\nb{V}$
	$$
\begin{aligned}
                &\,\,\;\;\;\, \cm{\case{e}{x_1}{e_1}{x_2}{e_2}}(\cm{\nv}) \\[.2em]
		        &=\cm{e_1}(\cm{e}_\rd{1})\rp\cm{e_2}(\cm{e}_\rd{2}) && \t{By compiling}\\[.2em]
&=\cm{e_1}\Bigl(\Bigl(\rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\cm{\vss{12}}\Bigr)_\rd{1}\Bigr)\rp\cm{e_2}\Bigl(\Bigl(\rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\cm{\vss{12}}\Bigr)_\rd{2}\Bigr) && \t{By induction}\\[.2em]
&=\cm{e_1}\Bigl(\rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\cm{\vss{12}}_\rd{1}\Bigr)\rp\cm{e_2}\Bigl(\rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\cm{\vss{12}}_\rd{2}\Bigr) && \t{Because first and second projection are linear}\\[.2em]
&=\Bigl(\rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\cm{e_1}(\cm{\vss{12}}_\rd{1})\Bigr) \rp \Bigl(\rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\cm{e_2}(\cm{\vss{12}}_\rd{2})\Bigr) && \t{Because }\cm{e_i} \t{ is linear}\\[.2em]
&=\rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\Bigl(\cm{e_1}(\cm{\vss{12}}_\rd{1}) \rp \cm{e_2}(\cm{\vss{12}}_\rd{2})\Bigr) && \t{Because }\rsum_\rd{a}\,\rd{f}(\rd{a})\rp\rsum_{\rd{a}}\,\rd{g}(\rd{a})=\rsum_{\rd{a}}\Bigl(\rd{f}(\rd{a})\rp\rd{g}(\rd{a})\Bigr)\\[.2em]
\end{aligned}
$$

	For $(\ders{12},\vss{12})\in\Vss{12}$, by inversion on $\ders{12}$ we know $\vss{12}$ is either $\error$, $\inj{1}{v_{22}}$, or $\inj{2}{v_{22}}$.
	- **Case** $\vss{12}=\inj{1}{\vss{22}}$
		Let $\Vss{22}=\{(\ders{22}, \vs) \mid (\_,\inj{1}{v_{22}}) \in \Vss{12},\,\nb{\{x_1 \map \vss{22}\}(\nv(e_1))} \bstep \vs\}$ 
		Observe that $(\nb{D},\vs) \in \Vs \iff (\nb{\ders{12}},\vss{12})\in \Vss{12} \;\land\; (\nb{\ders{22}},\vs)\in \Vss{22}$ when $\nb{\ders{12}}$ and $\ders{22}$ are obtained by inversion on $\nb{D}$
	$$
\begin{aligned}
                &\,\,\;\;\;\, \rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\Bigl(\cm{e_1}(\cm{\inj{1}{\vss{22}}}_\rd{1}) \rp \cm{e_2}(\cm{\inj{1}{\vss{22}}}_\rd{2})\Bigr) && \t{Because }\rsum_\rd{a}\,\rd{f}(\rd{a})\rp\rsum_{\rd{a}}\,\rd{g}(\rd{a})=\rsum_{\rd{a}}\Bigl(\rd{f}(\rd{a})\rp\rd{g}(\rd{a})\Bigr)\\[.2em]
                &= \rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\Bigl(\cm{e_1}({\rd{(}\cm{v_{22}}\rd{,0)_1}}) \rp \cm{e_2}(\cm{e_1}({\rd{(}\cm{v_{22}}\rd{,0)_2}})\Bigr) && \t{By compiling}\\[.2em]
	            &= \rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\Bigl(\cm{e_1}(\cm{v_{22}}) \rp \cm{e_2}(\cm{e_1}(\rd{0}))\Bigr) && \t{By first and second projection}\\[.2em]
	            &= \rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\cm{e_1}(\cm{v_{22}}) && \t{By linearity }\cm{e_i}(\rd{0})=\rd{0}\\[.2em]
	            &= \rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\cm{\{x_1 \map v_{22}\}(\nv(e_1))} && \t{Because compiling commutes with substitution}\\[.2em]
	            &= \rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\rsum_{(\ders{22},\,\vs)\in\Vss{22}}\cm{v} && \t{By induction}\\[.2em]
	            &= \rsum_{(\der,\,\vs)\in\Vs}\cm{v} && \t{Because }(\nb{D},\vs) \in \Vs \iff (\nb{\ders{12}},\vss{12})\in \Vss{12} \;\land\; (\nb{\ders{22}},\vs)\in \Vss{22}\\[.2em]
\end{aligned}
$$

	- **Case** $\vss{12}=\inj{2}{\vss{22}}$
		Similar to previous case
	- **Case** $\vss{12}=\error$ 
		Similar to previous case


- **Case** $\p{\D}{\pi_1e}{\ty_1}$
	**Show** $\cm{\pi_1e}(\cm{\nv})=\rsum_{(\der,\,\vss1)\in\Vs}\cm{v_1}$
	By assumption $\Vs=\{(\der,\vs) \mid \nv(\nb{\pi_1e}) \bstep \vs\}$
	Let $\nb{\Vss{12}}=\{(\nb{\ders{12}},\vss{12}) \mid \nv(\es) \bstep_\nb{\ders{1}}\vss{12}\}$ by inversion on derivations in $\nb{V}$
	$$
\begin{aligned}
                &\,\,\;\;\;\, \cm{\pi_1e}(\cm{\nv}) \\[.2em]
		        &=\cm{e}_\rd{1} && \t{By compiling}\\[.2em]
		        &=\Bigl(\rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\cm{\vss{12}}\Bigr)_\rd{1} && \t{By induction}\\[.2em]
		        &=\rsum_{(\ders{12},\,\vss{12})\in\Vss{12}}\cm{\vss{12}}_\rd{1} && \t{Because }\cm{v_{12}}_\rd{1} \t{ is linear}\\[.2em]
\end{aligned}
$$
	For $(\ders{12},\vss{12})\in\Vss{12}$, by inversion on $\ders{12}$ we know $\vss{12}$ is either $\nb{(e_1,e_2)}$ or $\error$.
	- **Case** $\vss{12}=\nb{(e_1,e_2)}$
		Let $\Vss{22}=\{(\ders{22}, \vs) \mid (\_,\nb{(e_1,e_2)}) \in \Vss{12},\,\nv(\ess1) \bstep \vs\}$ 
		Observe that $(\nb{D},\vs) \in \Vs \iff (\nb{\ders{12}},\nb{(e_1,e_2)})\in \Vss{12} \;\land\; (\nb{\ders{22}},\vs)\in \Vss{22}$ when $\nb{\ders{12}}$ and $\ders{22}$ are obtained by inversion on $\nb{D}$
$$
\begin{aligned}
                &\,\,\;\;\;\, \rsum_{(\ders{12},\,\nb{(e_1,e_2)})\in\Vss{12}}\cm{(e_1,e_2)}_\rd{1} \\[.2em]
		        &=\rsum_{(\ders{12},\,\nb{(e_1,e_2)})\in\Vss{12}}\rd{(}\cm{e_1}\rd{,}\cm{e_2}\rd{)}_\rd{1} && \t{By compiling}\\[.2em]
		        &=\rsum_{(\ders{12},\,\nb{(e_1,e_2)})\in\Vss{12}}\cm{e_1} && \t{By first projection}\\[.2em]
		        &=\rsum_{(\ders{12},\,\nb{(e_1,e_2)})\in\Vss{12}}\rsum_{(\ders{22},\,\vs)\in\Vss{22}}\cm{v} && \t{By induction}\\[.2em]
		        &=\rsum_{(\der,\,\vs)\in\Vs}\cm{v} && \t{Because }(\nb{D},\vs) \in \Vs \iff (\nb{\ders{12}},\nb{(e_1,e_2)})\in \Vss{12} \;\land\; (\nb{\ders{22}},\vs)\in \Vss{22}\\[.2em]
\end{aligned}
$$
	- **Case** $\vss{12}=\error$
		Similar to previous case
- **Case** $\p{\D}{\pi_2e}{\ty_2}$
	Similar to previous case
- **Case** $\p{\D}{e_1 \sqcap e_2}{\ty}$
	**Show** $\cm{e_1\sqcap e_2}(\cm{\nv})=\rsum_{(\der,\,\vs)\in\Vs}\cm{v}$ 
	By assumption $\Vs=\{(\der,\vs) \mid \nv(\nb{e_1\sqcap e_2}) \bstep_\der \vs \}$
	Let $\nb{\Vss1}=\{(\nb{\ders{1}},\vs) \mid \nv(\ess1) \bstep_\nb{\ders{1}}\vs\}$ by inversion on derivations in $\nb{V}$
	Let $\nb{\Vss2}=\{(\nb{\ders{2}},\vs) \mid \nv(\ess2) \bstep_\nb{\ders{2}}\vs\}$ by inversion on derivations in $\nb{V}$
	Observe that $(\der,\vs) \in\Vs \iff (\ders1,\vs)\in\Vss1 \;\lor\; (\ders2,\vs) \in \Vss2$ when $\ders1$ and $\ders2$ are obtained by inversion on $\der$
$$
\begin{aligned}
                &\,\,\;\;\;\, \cm{e_1 \sqcap e_2}(\cm{\nv}) \\[.2em]
		        &= \cm{e_1}\rp\cm{e_2} && \t{By compiling}\\[.2em]
		        &= \rsum_{(\ders1,\,\vs)\in\Vss1}\cm{v}\rp\rsum_{(\ders2,\,\vs)\in\Vss2}\cm{v} && \t{By induction}\\[.2em]
		        &= \rsum_{(\_,\,\vs)\in\Vss1 \cup\Vss2}\cm{v} && \t{Because }\rsum_{\rd{a} \in\rd{A}}\rd{a}\rp\rsum_{\rd{b} \in\rd{B}}\rd{b}=\rsum_{\rd{c}\in\rd{A \cup B}}\rd{c}\\[.2em]
		        &= \rsum_{(\der,\,\vs)\in\Vs}\cm{v} && \t{Because }(\der,\vs) \in\Vs \iff (\ders1,\vs)\in\Vss1 \;\lor\; (\ders2,\vs) \in \Vss2\\[.2em]
\end{aligned}
$$
- **Case** $\p{\D_\t{k}\o\D_\t{v}}{e_\t{kv}(e_\t{q})_\mathcal{R}}{\ty_\t{v}}$ 
	**Show** $\cm{e_\t{kv}(e_\t{q})_\mathcal{R}}(\cm{\nv})=\rsum_{(\der,\,\vs)\in\Vs}\cm{v}$ 
	By assumption $\Vs=\{(\der,\vs) \mid \nv(\nb{e_\t{kv}}\nb{(e_\t{q})_\mathcal{R}}) \bstep_\der \vs \}$ 
	Let $\nb{\Vss{\t{q}}}=\{(\nb{\ders{\t{q}}},\vss{\t{q}}) \mid \nv(\ess{\t{q}}) \bstep_\nb{\ders{\t{q}}}\vss{\t{q}}\}$ by inversion on derivations in $\nb{V}$
	Let $\nb{\Vss{\t{kv}}}=\{(\nb{\ders{\t{kv}}},\vss{\t{kv}}) \mid \nv(\ess{\t{kv}}) \bstep_\nb{\ders{\t{kv}}}\vss{\t{kv}}\}$ by inversion on derivations in $\nb{V}$

$$
\begin{aligned}
                &\,\,\;\;\;\, \cm{e_\t{kv}\nb{(e_\t{q})_\mathcal{R}}}(\cm{\nv}) \\[.2em]
		        &= \cm{e_\t{kv}}(\cm{\mathcal{R}}(\cm{e_\t{q}})) && \t{By compiling}\\[.2em]
		        &= \cm{e_\t{kv}}(\cm{\mathcal{R}}(\rsum_{(\ders{\t{q}},\vss{\t{q}})\in\Vss{\t{q}}}\cm{v_\t{q}})) && \t{By induction}\\[.2em]
		        &= \rsum_{(\ders{\t{q}},\vss{\t{q}})\in\Vss{\t{q}}}\cm{e_\t{kv}}(\cm{\mathcal{R}}(\cm{v_\t{q}})) && \t{Because }\cm{\mathcal{R}}\t{ and }\cm{e_\t{kv}}\t{ are linear}\\[.2em]
\end{aligned}
$$
	
	For $(\ders{\t{kv}},\vss{\t{kv}})\in\Vss{\t{kv}}$, by inversion on $\ders{\t{kv}}$ we know $\vss{\t{kv}}$ is either $\nb{\dict{v_\t{k}}{v_\t{v}}}$ or $\error$.
	- **Case** $\vss{\t{kv}}=\dict{v_\t{k}}{v_\t{v}}$
		Let $\nb{\Vss{\t{r}}}=\{(\nb{\ders{\t{r}}},\vs) \mid (\_,\vss{\t{q}}) \in \Vss{\t{q}},\, (\_, \dict{v_\t{k}}{v_\t{v}}) \in \Vss{\t{kv}},\,{\Large \nb{\sqcap}}_{(\nb{(v_\t{k})_i},\vss{\t{q}})\in\nb{\mathcal{R}}}\nb{(v_\t{v})_i} \bstep_\nb{\ders{\t{r}}}\vs\}$ by inversion on derivations in $\nb{V}$
		Observe that $(\der,\vs) \in\Vs \iff (\ders{\t{q}},\vss{\t{q}}) \in \Vss{\t{q}} \;\land\;(\nb{\ders{\t{kv}}},\dict{v_\t{k}}{v_\t{v}})\in\Vss{\t{kv}} \;\land\; (\ders{\t{r}},\vs) \in \Vss{\t{r}}$ when $\ders{\t{kv}}$ and $\ders{\t{kv}}$ and $\ders{\t{r}}$ are obtained by inversion on $\der$
$$
\begin{aligned}
                &\,\,\;\;\;\, \rsum_{(\ders{\t{q}},\vss{\t{q}})\in\Vss{\t{q}}}\cm{\dict{e_\t{k}}{e_\t{v}}}(\cm{\mathcal{R}}(\cm{v_\t{q}})) \\[.2em]
                &= \rsum_{(\ders{\t{q}},\vss{\t{q}})\in\Vss{\t{q}}}\Bigl(\rsum_{(\ders{\t{kv}},\dict{v_\t{k}}{v_\t{v}})\in\Vss{\t{kv}}}\cm{\dict{v_\t{k}}{v_\t{v}}}\Bigr)(\cm{\mathcal{R}}(\cm{v_\t{q}})) && \t{By induction}\\[.2em]
                &= \rsum_{(\ders{\t{q}},\vss{\t{q}})\in\Vss{\t{q}}}\rsum_{(\ders{\t{kv}},\dict{v_\t{k}}{v_\t{v}})\in\Vss{\t{kv}}}(\cm{\dict{v_\t{k}}{v_\t{v}}})(\cm{\mathcal{R}}(\cm{v_\t{q}})) && \t{Because }\rd{\sum}_\rd{a} (\rd{f_1}(\rd{a})\rp\rd{f_2}(\rd{a}))=\rd{\sum}_\rd{a}\rd{\sum}_\rd{i}\,\rd{f}_\rd{i}(\rd{a}) \\[.2em]
		        &= \rsum_{(\ders{\t{q}},\vss{\t{q}})\in\Vss{\t{q}}}\rsum_{(\ders{\t{kv}},\dict{v_\t{k}}{v_\t{v}})\in\Vss{\t{kv}}}\Bigl(\xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\,\rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\vss{\t{k}}}_\rd{i}\rd{,\,\xvec_\t{q}\rangle} \rt \cm{\vss{\t{v}}}_\rd{i}\Bigr)(\cm{\mathcal{R}}(\cm{v_\t{q}})) && \t{By compiling}\\[.2em]
		        &= \rsum_{(\ders{\t{q}},\vss{\t{q}})\in\Vss{\t{q}}}\rsum_{(\ders{\t{kv}},\dict{v_\t{k}}{v_\t{v}})\in\Vss{\t{kv}}}\Bigl(\rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\vss{\t{k}}}_\rd{i}\rd{,}\,\cm{\mathcal{R}}(\cm{v_\t{q}})\rd{\rangle} \rt \cm{\vss{\t{v}}}_\rd{i}\Bigr) && \t{By function application}\\[.2em]
		        &= \rsum_{(\ders{\t{q}},\vss{\t{q}})\in\Vss{\t{q}}}\rsum_{(\ders{\t{kv}},\dict{v_\t{k}}{v_\t{v}})\in\Vss{\t{kv}}}\Bigl(\rsum_{(\nb{(v_\t{k})_i},\vss{\t{q}})\in\nb{\mathcal{R}}}\cm{v_\t{v}}_\rd{i}\Bigr) && \t{Because }\nb{\mathcal{R}} \t{ is implemented by }\rd{\langle\_,\_\rangle}\\[.2em]
		        &= \rsum_{(\ders{\t{q}},\vss{\t{q}})\in\Vss{\t{q}}}\rsum_{(\ders{\t{kv}},\dict{v_\t{k}}{v_\t{v}})\in\Vss{\t{kv}}}\Bigl(\rsum_{(\nb{(v_\t{k})_i},\vss{\t{q}})\in\nb{\mathcal{R}}}\cm{(v_\t{v})_\nb{i}}\Bigr) && \t{By compiling}\\[.2em]
		        &= \rsum_{(\ders{\t{q}},\vss{\t{q}})\in\Vss{\t{q}}}\rsum_{(\ders{\t{kv}},\dict{v_\t{k}}{v_\t{v}})\in\Vss{\t{kv}}}\Bigl(\cm{{\Large \sqcap}_{(\nb{(v_\t{k})_i},\vss{\t{q}})\in\nb{\mathcal{R}}}(v_\t{v})_i}\Bigr) && \t{By compiling}\\[.2em]
		        &= \rsum_{(\ders{\t{q}},\vss{\t{q}})\in\Vss{\t{q}}}\rsum_{(\ders{\t{kv}},\dict{v_\t{k}}{v_\t{v}})\in\Vss{\t{kv}}}\Bigl(\rsum_{(\ders{\t{r}},\,\vs)\in\Vss{\t{r}}}\cm{\vs}\Bigr) && \t{By induction}\\[.2em]
		        &= \rsum_{(\der,\,\vs)\in\Vs}\cm{\vs} && \t{Because }(\der,\vs) \in\Vs \iff \cdots \;\land\; (\ders{\t{r}},\vs) \in \Vss{\t{r}}\\[.2em]
\end{aligned}
$$
	- **Case** $\vss{\t{kv}}=\error$
		Similar to previous case
$$\blacksquare$$


###### Lemma ($\nb{\mathcal{R}}$ is implemented by $\rd{\langle\_,\_\rangle}$) 
If $(\vss{\t{k}})_\nb{i}:\nb{\t{enum}}(\nb{\ty_{\t{k}}})$ and $\vss{\t{q}}:\nb{\t{enum}}(\nb{\ty_{\t{q}}})$, then
- $((\vss{\t{k}})_\nb{i},\vss{\t{q}}) \in\nb{\mathcal{R}} \implies \rd{\langle}\cm{v_\t{k}}_\rd{i}\rd{,}\,\cm{\mathcal{R}}(\cm{v_\t{q}})\rd{\rangle}\rt\cm{v_\t{v}}_\rd{i}=\cm{v_\t{v}}_\rd{i}$  
- $((\vss{\t{k}})_\nb{i},\vss{\t{q}}) \not\in\nb{\mathcal{R}} \implies \rd{\langle}\cm{v_\t{k}}_\rd{i}\rd{,}\,\cm{\mathcal{R}}(\cm{v_\t{q}})\rd{\rangle}=\rd{0}$

**Proof.** Consider each implication,
- **Case** $((\vss{\t{k}})_\nb{i},\vss{\t{q}}) \in\nb{\mathcal{R}} \implies \rd{\langle}\cm{v_\t{k}}_\rd{i},\cm{R}(\cm{v_\t{q}})\rd{\rangle}\rt\cm{v_\t{v}}_\rd{i}=\cm{v_\t{v}}_\rd{i}$
	By assumption $((\vss{\t{k}})_\nb{i},\vss{\t{q}}) \in \nb{\mathcal{R}}$
	Because $(\vss{\t{k}})_\nb{i}:\nb{\t{enum}}(\nb{\ty_{\t{k}}})$, $\cm{(v_\t{k})_i}$ is a one-hot vector, where $\cm{(v_\t{k})_i}_\rd{j}=\rd{1}$
	Since all other entries of $\cm{(v_\t{k})_i}_\rd{j}$ are $\rd{0}$, $\rd{\langle}\cm{v_\t{k}}_\rd{i},\cm{R}(\cm{v_\t{q}})\rd{\rangle}=\rd{\langle}\cm{(v_\t{k})_\nb{i}},\cm{R}(\cm{v_\t{q}})\rd{\rangle}=\cm{(v_k)_\nb{i}}_\rd{j}\rt\cm{\mathcal{R}}(\cm{v_\t{q}})_\rd{j}$  
	Observe that $\cm{\mathcal{R}}(\cm{v_\t{q}})_\rd{j}=\rd{1}$ when $((\vss{\t{k}})_\nb{i},\vss{\t{q}}) \in \nb{\mathcal{R}}$
$$
\begin{aligned}
                &\,\,\;\;\;\, \rd{\langle}\cm{v_\t{k}}_\rd{i},\cm{R}(\cm{v_\t{q}})\rd{\rangle}\rt\cm{v_\t{v}}_\rd{i} \\[.2em]
                &= \rd{\langle}\cm{(v_\t{k})_\nb{i}},\cm{R}(\cm{v_\t{q}})\rd{\rangle}\rt\cm{v_\t{v}}_\rd{i}  && \t{By compiling}\\
                &= \cm{(v_k)_\nb{i}}_\rd{j}\rt\cm{\mathcal{R}}(\cm{v_\t{q}})_\rd{j}\rt\cm{v_\t{v}}_\rd{i}  && \t{Because }\cm{(v_\t{k})_i} \t{ is one-hot}\\[.2em]
                &= \rd{1}\rt\cm{\mathcal{R}}(\cm{v_\t{q}})_\rd{j}\rt\cm{v_\t{v}}_\rd{i}  && \t{Because }\cm{(v_\t{k})_i}_\rd{j}=\rd{1}\\[.2em]
                &= \rd{1}\rt\rd{1}\rt\cm{v_\t{v}}_\rd{i}  && \t{Because }\cm{\mathcal{R}}(\cm{v_\t{q}})_\rd{j}=\rd{1}\\[.2em]
                &= \cm{v_\t{v}}_\rd{i}  && \t{By multiplication}\\[.2em]
\end{aligned}
$$
- **Case** $((\vss{\t{k}})_\nb{i},\vss{\t{q}}) \not\in\nb{\mathcal{R}} \implies \rd{\langle}\cm{v_\t{k}}_\rd{i}\rd{,}\,\cm{\mathcal{R}}(\cm{v_\t{q}})\rd{\rangle}=\rd{0}$
	Similar to previous case but using $\cm{\mathcal{R}}(\cm{v_\t{q}})_\rd{j}=\rd{0}$ when $((\vss{\t{k}})_\nb{i},\vss{\t{q}}) \notin \nb{\mathcal{R}}$
$$\blacksquare$$

###### Lemma (Compiler commutes with substitution)
If $\p{\D,\bind{x}{\ty_1}}{e}{\ty_2}$,
$$\cm{\{x \map v\}(e)}(\svec)=\cm{e}(\svec,\cm{v})$$
**Proof.** By induction on $\es$. The argument follows a similar structure to the substitution lemma proved in (Velez-Ginorio et al., 2026).
$$\blacksquare$$


###### Lemma (Compiler maps programs to multilinear maps)
If $\p{\D,\bind{x}{\ty_1}}{e}{\ty_2}$,
$$\cm{e}(\svec,\a1\rt\xv1\rp\a2\rt\xv2)=\a1\rt\cm{e}(\svec,\xv1)\rp \a2\rt\cm{e}(\svec,\xv2)$$
**Proof.** Immediate from compiler mapping programs to additive and homogeneous maps.
$$\blacksquare$$
###### Lemma (Compiler maps programs to additive maps)
If $\p{\D,\bind{x}{\ty_1}}{e}{\ty_2}$,
$$\cm{e}(\svec,\xv1\rp\xv2)=\cm{e}(\svec,\xv1)\rp\cm{e}(\svec,\xv2)$$
 **Proof.** By induction on $\nb{e}$,
- **Case** $\ys$
	Consider whether $\xs=\ys$,
	- **Case** $\xs=\ys$
		By inversion $\nb{\D}=\nb{\emp}$ 
		$$
		\begin{aligned}
			&\;\;\;\;\;\cm{x}(\xvec\rp\yvec) && \\
			&= \xvec \rp \yvec && \t{By compiling}\\
			&= \cm{x}(\xvec)\rp\cm{y}(\yvec) && \t{By compiling}
		\end{aligned}
		$$
	- **Case** $\xs \neq \ys$
		Vacuous because we assume a contradiction $\p{\D,\bind{x}{\ty_1}}{y}{\ty_2}$ 
- **Case** $\unit$
	Vacuous because we assume $\p{\D,\bind{x}{\ty_1}}{\unit}{\ty_2}$ 
- **Case** $\error$
		$$
		\begin{aligned}
			&\;\;\;\;\;\cm{\error}(\svec,\xv1\rp\xv2) && \\
			&= \rd{0} && \t{By compiling}\\
			&= \rd{0} \rp \rd{0} && \t{Because }\rd{0+0=0}\\
			&= \cm{\error}(\svec,\xv1)\rp\cm{\error}(\svec,\xv2) && \t{By compiling}
		\end{aligned}
		$$
- **Case** $\inj{1}{e}$
		$$
		\begin{aligned}
			&\;\;\;\;\;\cm{\inj{1}{e}}(\svec,\xv1\rp\xv2) && \\
			&= \rd{(}\cm{e}(\svec,\xv1\rp\xv2)\rd{, 0)} && \t{By compiling}\\
			&= \rd{(}\cm{e}(\svec,\xv1)\rp\cm{e}(\svec,\xv2)\rd{, 0)} && \t{By induction}\\
			&= \rd{(}\cm{e}(\svec,\xv1)\rd{, 0)}\rp\rd{(}\cm{e}(\svec,\xv2)\rd{, 0)} && \t{Because }\rd{(a+b,0)}=\rd{(a,0)\rp(b,0)}\\
			&= \cm{\inj{1}{e}}(\svec,\xv1)\rp\cm{\inj{1}{e}}(\svec,\xv2) && \t{By compiling}
		\end{aligned}
		$$

- **Case** $\inj{2}{e}$
	Similar to previous case
- **Case** $\nb{(e_1,e_2)}$
		$$
		\begin{aligned}
			&\;\;\;\;\;\cm{(e_1,e_2)}(\svec,\xv1\rp\xv2) && \\
			&= \rd{(}\cm{e_1}(\svec,\xv1\rp\xv2)\rd{,} \,\cm{e_2}(\svec,\xv1\rp\xv2)\rd{)} && \t{By compiling}\\
			&= \rd{(}\cm{e_1}(\svec,\xv1)\rp\cm{e_1}(\svec,\xv2)\rd{,} \,\cm{e_2}(\svec,\xv1)\rp\cm{e_2}(\svec,\xv2)\rd{)} && \t{By induction}\\
			&=\rd{(}\cm{e_1}(\svec,\xv1)\rd{,} \,\cm{e_2}(\svec,\xv1)\rd{)}\rp\rd{(}\cm{e_1}(\svec,\xv2)\rd{,} \,\cm{e_2}(\svec,\xv2)\rd{)} && \t{Because }\rd{(a+b,c+d)}=\rd{(a,c)+(b,d)}\\
			&= \cm{(e_1,e_2)}(\svec,\xv1)\rp\cm{(e_1,e_2)}(\svec,\xv2) && \t{By compiling}
		\end{aligned}
		$$
- **Case** $\dict{e_\t{k}}{e_\t{v}}$
	Consider whether $\xs \in\fv{\ess{\t{k}}}$ or $\xs \in\fv{\ess{\t{v}}}$,
	- **Case** $\xs \in\fv{\ess{\t{k}}}$
		$$
		\begin{aligned}
			&\;\;\;\;\;\cm{\dict{e_\t{k}}{e_\t{v}}}(\svec,\xv1\rp\xv2) && \\[.2em]
			&= \xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\,\rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\ess{\t{k}}}(\sv{\t{k}},\xv1\rp\xv2)_\rd{i}\rd{,\,\xvec_\t{q}\rangle} \rt \cm{\ess{\t{v}}}(\sv{\t{v}})_\rd{i} && \t{By compiling}\\[.2em]
			&= \xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\,\rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\ess{\t{k}}}(\sv{\t{k}},\xv1)_\rd{i}\rp\cm{\ess{\t{k}}}(\sv{\t{k}},\xv2)_\rd{i}\rd{,\,\xvec_\t{q}\rangle} \rt \cm{\ess{\t{v}}}(\sv{\t{v}})_\rd{i} && \t{By induction}\\[.2em]
			&= \xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\,\rsum_\rd{i=1}^\nb{n}\,\rd{(}\rd{\langle}\cm{\ess{\t{k}}}(\sv{\t{k}},\xv1)_\rd{i}\rd{,\,\xvec_\t{q}\rangle}\rp\rd{\langle}\cm{\ess{\t{k}}}(\sv{\t{k}},\xv2)_\rd{i}\rd{,\,\xvec_\t{q}\rangle}\rd{)} \rt \cm{\ess{\t{v}}}(\sv{\t{v}})_\rd{i} && \t{Because inner product is bilinear}\\[.2em]
			&= \xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\,\rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\ess{\t{k}}}(\sv{\t{k}},\xv1)_\rd{i}\rd{,\,\xvec_\t{q}\rangle}\rt \cm{\ess{\t{v}}}(\sv{\t{v}})_\rd{i} \rp \rd{\langle}\cm{\ess{\t{k}}}(\sv{\t{k}},\xv2)_\rd{i}\rd{,\,\xvec_\t{q}\rangle} \rt \cm{\ess{\t{v}}}(\sv{\t{v}})_\rd{i} && \t{Because }\rd{\cdot} \t{ distributes over }\rp\\[.2em]
			&= \xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\,\rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\ess{\t{k}}}(\sv{\t{k}},\xv1)_\rd{i}\rd{,\,\xvec_\t{q}\rangle}\rt \cm{\ess{\t{v}}}(\sv{\t{v}})_\rd{i} \\[.2em]
			&\;\;\;\;\;\;\;\;\;\,\,\rp \rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\ess{\t{k}}}(\sv{\t{k}},\xv2)_\rd{i}\rd{,\,\xvec_\t{q}\rangle} \rt \cm{\ess{\t{v}}}(\sv{\t{v}})_\rd{i} && \t{Because }\rsum_\rd{a}\,\rd{f}(\rd{a})\rp\rd{f}(\rd{b})=\rsum_\rd{a}\,\rd{f}(\rd{a})\rp\rsum_\rd{a}\,\rd{f}(\rd{b})\\[.2em]
			&= \Bigl(\xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\,\rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\ess{\t{k}}}(\sv{\t{k}},\xv1)_\rd{i}\rd{,\,\xvec_\t{q}\rangle}\rt \cm{\ess{\t{v}}}(\sv{\t{v}})_\rd{i}\Bigr) \\[.2em]
			&\rp\,\Bigl(\xvec_\rd{\t{q}} \,\,\rd{\mapsto}\,\, \rsum_\rd{i=1}^\nb{n}\,\rd{\langle}\cm{\ess{\t{k}}}(\sv{\t{k}},\xv2)_\rd{i}\rd{,\,\xvec_\t{q}\rangle} \rt \cm{\ess{\t{v}}}(\sv{\t{v}})_\rd{i}\Bigr) && \t{By addition of functions}\\[.2em]
			&= \cm{\dict{e_\t{k}}{e_\t{v}}}(\svec,\xv1)\rp\cm{\dict{e_\t{k}}{e_\t{v}}}(\svec,\xv2) && \t{By compiling}
		\end{aligned}
		$$
	- **Case** $\xs \in\fv{\ess{\t{v}}}$
		Similar to previous case

- **Case** $\nb{\pi_ie}$
		$$
		\begin{aligned}
			&\;\;\;\;\;\cm{\pi_ie}(\svec,\xv1\rp\xv2) && \\
			&= \cm{e}(\svec,\xv1\rp\xv2)_\rd{i} && \t{By compiling}\\
			&= \cm{e}(\svec,\xv1)_\rd{i}\rp\cm{e}(\svec,\xv2)_\rd{i} && \t{By induction}\\
			&= \cm{\pi_ie}(\svec,\xv1)\rp\cm{\pi_ie}(\svec,\xv2) && \t{By compiling}
		\end{aligned}
		$$
- **Case** $\nb{\case{e}{x_1}{e_1}{x_2}{e_2}}$
	Consider whether $\xs \in\fv{\es}$ or $(\xs \in\fv{\ess{1}}) \land (\xs \in \fv{\ess2})$,
	- **Case** $\xs \in\fv{\es}$
		$$
		\begin{aligned}
			&\;\;\;\;\;\cm{\nb{\case{e}{x_1}{e_1}{x_2}{e_2}}}(\svec,\xv1\rp\xv2) && \\
			&= \cm{e_1}(\sv2,\cm{e}(\sv1,\xv1\rp\xv2)_\rd{1})\rp\cm{e_2}(\sv2,\cm{e}(\sv1,\xv1\rp\xv2)_\rd{2}) && \t{By compiling}\\
			&= \cm{e_1}(\sv2,\cm{e}(\sv1,\xv1)_\rd{1}\rp\cm{e}(\sv1,\xv2)_\rd{1})\rp\cm{e_2}(\sv2,\cm{e}(\sv1,\xv1)_\rd{2}\rp\cm{e}(\sv1,\xv2)_\rd{2}) && \t{By induction and linearity of projection}\\
			&= \cm{e_1}(\sv2,\cm{e}(\sv1,\xv1)_\rd{1})\rp\cm{e_1}(\sv2,\cm{e}(\sv1,\xv2)_\rd{1})\\
			&\,\rp\cm{e_2}(\sv2,\cm{e}(\sv1,\xv1)_\rd{2})\rp\cm{e_2}(\sv2,\cm{e}(\sv1,\xv2)_\rd{2}) && \t{By induction}\\
			&= \cm{e_1}(\sv2,\cm{e}(\sv1,\xv1)_\rd{1})\rp\cm{e_2}(\sv2,\cm{e}(\sv1,\xv1)_\rd{2})\\
			&\,\rp\cm{e_1}(\sv2,\cm{e}(\sv1,\xv2)_\rd{1})\rp\cm{e_2}(\sv2,\cm{e}(\sv1,\xv2)_\rd{2}) && \t{Because}\rp \t{is commutative}\\
			&= \cm{\nb{\case{e}{x_1}{e_1}{x_2}{e_2}}}(\svec,\xv1)\\
			&\,\rp\cm{\nb{\case{e}{x_1}{e_1}{x_2}{e_2}}}(\svec,\xv2) && \t{By compiling}
		\end{aligned}
		$$
	- **Case** $(\xs \in\fv{\ess{1}}) \land (\xs \in \fv{\ess2})$
		Similar to previous case

- **Case** $\let{x}{e_1}{e_2}$
	Consider whether $\xs \in\fv{\ess1}$ or $\xs \in \fv{e_2}$,
	- **Case** $\xs \in \fv{\ess1}$
		$$
		\begin{aligned}
			&\;\;\;\;\;\cm{\let{x}{e_1}{e_2}}(\svec,\xv1\rp\xv2) && \\
			&= \cm{e_2}(\sv2,\cm{e_1}(\sv1,\xv1\rp\xv2)) && \t{By compiling}\\
			&= \cm{e_2}(\sv2,\cm{e_1}(\sv1,\xv1)\rp\cm{e_1}(\sv1,\xv2)) && \t{By induction}\\
			&= \cm{e_2}(\sv2,\cm{e_1}(\sv1,\xv1))\rp\cm{e_2}(\cm{e_1}(\sv1,\xv2)) && \t{By induction}\\
			&= \cm{\let{x}{e_1}{e_2}}(\svec,\xv1)\rp\cm{\let{x}{e_1}{e_2}}(\svec,\xv2) && \t{By compiling}
		\end{aligned}
		$$
	- **Case** $\xs \in \fv{e_2}$
		Similar to previous case
- **Case** $\nb{e_\t{kv}(e_\t{q})_\mathcal{R}}$
	Similar to previous case, but also using linearity of $\cm{\mathcal{R}}$
- **Case** $\nb{\ess1\sqcap\ess2}$
		$$
		\begin{aligned}
			&\;\;\;\;\;\cm{e_1\sqcap e_2}(\svec,\xv1\rp\xv2) && \\
			&= \cm{e_1}(\svec,\xv1\rp\xv2)\rp\cm{e_2}(\svec,\xv1\rp\xv2) && \t{By compiling}\\
			&= \cm{e_1}(\svec,\xv1)\rp\cm{e_1}(\svec,\xv2)\rp\cm{e_2}(\svec,\xv1)\rp\cm{e_2}(\svec,\xv2) && \t{By induction}\\
			&= \cm{e_1}(\svec,\xv1)\rp\cm{e_2}(\svec,\xv1)\rp\cm{e_1}(\svec,\xv2)\rp\cm{e_2}(\svec,\xv2) && \t{Because}\rp\t{is commutative}\\
			&= \cm{e_1\sqcap e_2}(\svec,\xv1)\rp\cm{e_1\sqcap e_2}(\svec,\xv2) && \t{By compiling}
		\end{aligned}
		$$

$$\blacksquare$$

###### Lemma (Compiler maps programs to homogeneous maps)
If $\p{\D,\bind{x}{\ty_1}}{e}{\ty_2}$,
$$\cm{e}(\svec,\rd{\alpha\.\xvec})=\rd{\alpha} \rt \cm{e}(\svec,\xvec)$$ **Proof.** By induction on $\es$. The argument follows a similar structure to proving that the compiler maps programs to additive maps.
$$\blacksquare$$

