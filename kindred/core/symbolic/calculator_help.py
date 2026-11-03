"""Help text shown in the Symbolic Calculator window."""
from __future__ import annotations


def calculator_help_html() -> str:
    return """
<h1>Symbolic Calculator</h1>
<p>Use Calculator to examine rates, differentiate expressions, solve equations, and substitute the
answers into further calculations. It uses the model in the current tab without changing it.
Independent parameters stay symbolic. Values from reaction defaults, simulations, or slider previews
are not inserted automatically.</p>

<h2>Calculate a Product Rate</h2>
<p>The examples follow this mechanism, with two intermediates X and Y leading to product P:</p>
<pre>A &lt;-&gt; X; kf=2; kr=3
X &lt;-&gt; Y; kf=5; kr=7
Y -&gt; P; k=11</pre>
<p>The examples calculate the product rate, examine its dependence on Y and k3, then solve for both
intermediates and use those answers to express the rate in terms of A. Later sections add signal
definitions and one reaction to this same model. Each addition is shown before it is used.</p>
<p>Enter one query and click <b>Evaluate</b>, or press <b>Enter</b>.
The tables show the mathematical results. Calculator may arrange equivalent terms differently
and also display notices or queries for reusing an answer.</p>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Task</th><th>Query</th><th>Mathematical Result</th></tr>
<tr><td>Get the product rate</td><td><code>dP/dt</code></td><td><code>Y*k3</code></td></tr>
<tr><td>Insert the rate constant</td><td><code>subs(dP/dt,k3,11)</code></td><td><code>11*Y</code></td></tr>
<tr><td>Evaluate at Y=0.1 and k3=11</td><td><code>subs(dP/dt,[Y,k3],[1/10,11])</code></td><td><code>11/10</code></td></tr>
</table>
<p>The third reaction supplies the name k3. Although the source says k=11, the first query keeps k3
symbolic. Each substitution applies only to the expression in that query. It does not set a value
for later queries or change the model.</p>
<p>For one usable answer, click <b>Use Result</b> to insert the complete query needed to reproduce it.
This does not evaluate the query. For example, after evaluating <code>dP/dt</code>, click
<b>Use Result</b>, change the inserted text to <code>subs(dP/dt,k3,11)</code>, then click
<b>Evaluate</b>.</p>
<p>Results use mathematical formatting to make fractions, equations, and matrices easier to read.
Scroll horizontally to read wide expressions. The displayed notation is not necessarily valid input.
Use <b>Use Result</b> or <b>Use in Query</b> to insert an executable query.</p>
<p><b>Copy Result</b> copies the text report with exact values, all evaluated answers,
notices, and any reuse queries. Selecting an answer does not narrow this report.
<b>Copy Context</b> adds the reaction source, definitions, and assumptions to the same text report.</p>

<h2>Units in Calculator Queries</h2>
<p>When substituting numerical values, use M for concentrations, s for time, K for temperature,
J/mol for energy parameters, and m<sup>3</sup> for volume.</p>
<p>For example, to substitute a concentration of 1 mM for Y in the example model, enter:</p>
<pre>subs(dP/dt,Y,1/1000)</pre>
<p>For a model whose first reaction is irreversible with an Arrhenius <code>Ea1</code>, enter 5000 to
substitute 5 kJ/mol once the rate constant is written out:</p>
<pre>subs(subs(k1,definitions()),Ea1,5000)</pre>
<p>These query units apply even when the corresponding slider or fitting field uses another input unit.</p>
<p>Results and <b>Copy Context</b> include known units. If Kindred cannot determine consistent units,
it leaves them unspecified and continues the calculation.</p>
<p>Use <b>Use Result</b> or <b>Use in Query</b> to reuse an answer with its unit conversions preserved.</p>
<h2>Names, Definitions, and Exact Values</h2>
<p>Queries can use species, parameters, and definitions from the model in the current tab.</p>
<p>For species A, <code>A</code> and <code>[A]</code> both represent its concentration at simulation
time <i>t</i>. <code>initial(A)</code> and <code>[A]_0</code> represent its symbolic initial concentration.
Calculator does not automatically insert the numerical value from an initial-condition set.</p>
<p><code>t</code> represents time in seconds and <code>T</code> temperature in kelvin.
Concentrations and initial concentrations are nonnegative. Rate constants, equilibrium constants,
Arrhenius prefactors, kappa, degeneracy and T are positive. Energies, time and other parameters are real.</p>
<p>On an Arrhenius or Eyring reaction, or an equilibrium given by <code>dG_eq</code>, whose energy is an
independent parameter, and whose prefactor, kappa and degeneracy do not depend on any such energy of
this or another step, the rate constant or
<code>Keq</code> is a symbol of its own, such as <code>k1</code> or
<code>Keq1</code>. Rates, steady states and other results are written in rate constants and can be
differentiated with respect to them, as in <code>partial(dA/dt,k1)</code>. The energy is written through
its rate constant: <code>Ea1</code> gives <code>-R*T*log(k1/A1)</code>, so an energy that a param line defines
from it, such as <code>Ea2 = Ea1 + offset</code>, relates k2 to k1. <code>definitions()</code> lists the
definitions of these rate constants, and <code>subs(F,definitions())</code> writes them out in energies,
prefactors and T. A calculation in T, an energy, a prefactor, kappa or degeneracy needs them written out
first, for example <code>solve(subs(eq(k1,k2),definitions()),T)</code> or
<code>partial(subs(dB/dt,definitions()),Ea1)</code>. Putting a number in for such a rate constant
fixes it, so a later calculation in T holds it fixed instead of following its temperature
dependence; for temperature dependence, write rate constants out first.</p>
<p>Use <code>+ - * / ^ **</code> for arithmetic and parentheses to group expressions.
Literal numbers remain exact. For example, <code>1/10+2/10</code> gives <code>3/10</code>.
Use <code>approx(F)</code> for a decimal approximation with 15 significant digits, or
<code>approx(F,digits)</code> to choose a positive integer number of digits.
For example, <code>approx(subs(dP/dt,[Y,k3],[1/10,11]),6)</code> gives a decimal approximation
of 1.1. Approximation does not supply missing parameter values.</p>
<p>In the reference forms below, F means an expression you supply, x means a variable, and M means
a matrix. Names such as result and solutions stand for a complete calculation, such as a call to
solve. The placeholders lhs and rhs are expressions, and equation1 and equation2 are equations
made with eq. These are not predefined names for previous results. Calculator queries do
not create assignments. In particular, <code>eq</code> constructs an equation rather than assigning
a value. Worked queries use the actual names from the example model.</p>
<p>Fixed constants are <code>k_B</code>, <code>h</code>, <code>h_bar</code>, <code>N_A</code>, and
<code>R</code>. The gas constant <i>R</i> uses J/(mol·K). Their values cannot be substituted or solved for.</p>

<h3>Use a Parameter Definition in a Signal Calculation</h3>
<p>Extend the three-reaction example with these definitions for a signal proportional to P:</p>
<pre>param gain=2
param response=2*gain
let signal=response*[P]</pre>
<p>Gain is an independent parameter. Its default is 2, but Calculator keeps it symbolic.
Response is defined in terms of gain, so Calculator replaces response with 2*gain.
The let name signal represents the resulting expression 2*gain*P.</p>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Query</th><th>Mathematical Result</th></tr>
<tr><td><code>signal</code></td><td><code>2*gain*P</code></td></tr>
<tr><td><code>partial(signal,gain)</code></td><td><code>2*P</code></td></tr>
<tr><td><code>subs(signal,gain,2)</code></td><td><code>4*P</code></td></tr>
</table>
<p>When choosing a variable to solve for, differentiate with respect to, integrate over, or substitute,
use an independent variable. Here, use gain rather than response or signal.
This requirement applies to <code>solve</code>, <code>partial</code>, <code>integrate</code>,
<code>limit</code>, <code>series</code>, <code>collect</code>, and three-argument <code>subs</code>.
The operations <code>initial</code> and <code>steady_state</code> take species names.
The remaining examples keep these signal definitions.</p>

<h2>Rates and Derivatives</h2>
<p>The following queries use the three-reaction mechanism and the signal definitions above.</p>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Write</th><th>Meaning</th></tr>
<tr><td><code>dP/dt</code></td><td>The rate for species P from the model. Replace P with another species name to get its rate.</td></tr>
<tr><td><code>dt(F)</code> or <code>d(F)/dt</code><br><code>dt(F,n)</code></td><td>The total kinetic derivative. It differentiates explicit dependence on t and adds the partial derivative with respect to each species concentration multiplied by the corresponding species rate. Supply n to repeat the operation.</td></tr>
<tr><td><code>partial(F,x)</code><br><code>partial(F,x,n)</code></td><td>The partial derivative with respect to independent variable x. Supply n to differentiate repeatedly with respect to x.</td></tr>
<tr><td><code>dX/dY</code></td><td>The trajectory slope <code>(dX/dt)/(dY/dt)</code>, simplified algebraically.</td></tr>
</table>
<p>For both dt and partial, n is a nonnegative integer and defaults to 1.
Independent parameters, initial concentrations, and T stay fixed during dt, even when they appear
in the expression.</p>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Task</th><th>Query</th><th>Mathematical Result</th></tr>
<tr><td>Differentiate the product rate with respect to Y</td><td><code>partial(dP/dt,Y)</code></td><td><code>k3</code></td></tr>
<tr><td>Differentiate the product rate with respect to k3</td><td><code>partial(dP/dt,k3)</code></td><td><code>Y</code></td></tr>
<tr><td>Differentiate the product rate along the model dynamics</td><td><code>dt(dP/dt)</code></td><td><code>k3*(kf2*X-(kr2+k3)*Y)</code></td></tr>
<tr><td>Differentiate the rate for A with respect to X</td><td><code>d(dA/dt)/dX</code></td><td><code>kr1</code>, also obtained with <code>partial(dA/dt,X)</code></td></tr>
<tr><td>Compare two independent concentration variables</td><td><code>partial(X,Y)</code></td><td><code>0</code></td></tr>
</table>
<p>To compare changes in X and Y along the trajectory, enter <code>dX/dY</code>. For this model it gives:</p>
<pre>(kf1*A-(kr1+kf2)*X+kr2*Y)/(kf2*X-(kr2+k3)*Y)</pre>
<p>This rate ratio is a different calculation from partial(X,Y).
For any model with species A and B, the notation <code>d(dA/dt)/dB</code> means the partial
derivative of the rate for A with respect to B. In this example, X takes the place of B.</p>

<h2>Substitution and Algebra</h2>
<p><code>subs(F,x,value)</code> replaces one independent variable.
Use <code>subs(F,[x,y],[a,b])</code> to replace several variables at once.
Here x and y are the targets, and a and b are their replacement expressions.
Targets must be distinct and the two lists must have the same length.</p>
<p>List substitution is simultaneous. Nested calls perform substitutions in sequence:</p>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Query</th><th>Mathematical Result</th></tr>
<tr><td><code>subs(X+Y,[X,Y],[Y,2])</code></td><td><code>Y+2</code>. The Y introduced by the first replacement is not replaced again.</td></tr>
<tr><td><code>subs(subs(X+Y,X,Y),Y,2)</code></td><td><code>4</code>. The outer call replaces both occurrences of Y.</td></tr>
</table>
<p>Use algebra operations to arrange an expression for the calculation you want to perform:</p>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Write</th><th>Purpose</th><th>Example</th></tr>
<tr><td><code>collect(F,x)</code></td><td>Group terms by powers of independent variable x.</td><td><code>collect(dX/dt,X)</code> gives <code>kf1*A+kr2*Y-(kf2+kr1)*X</code>, grouping the terms that consume X.</td></tr>
<tr><td><code>expand(F)</code></td><td>Expand products and powers.</td><td><code>expand(collect(dX/dt,X))</code> separates the two terms proportional to X again.</td></tr>
<tr><td><code>factor(F)</code></td><td>Factor an expression.</td><td><code>factor(kf2*X+kr1*X)</code> gives <code>X*(kf2+kr1)</code>.</td></tr>
<tr><td><code>cancel(F)</code></td><td>Cancel common polynomial factors in a fraction.</td><td><code>cancel((X^2-Y^2)/(X-Y))</code> gives <code>X+Y</code>.</td></tr>
<tr><td><code>simplify(F)</code></td><td>Apply general symbolic simplification.</td><td><code>simplify((X+Y)^2-X^2-2*X*Y)</code> gives <code>Y^2</code>.</td></tr>
</table>
<p>Simplification can change the form without preserving every restriction on the original
expression. For example, <code>X/X</code> and <code>subs(X/X,X,0)</code> both give <code>1</code>.
An expression already found to be undefined, such as <code>X/0</code> or <code>(X/0)^0</code>,
remains unavailable. An algebraically simplified formula does not prove that the original
calculation was defined at every value.</p>

<h2>Solve Equations and Eliminate Intermediates</h2>
<p>Use the three-reaction model with its signal definitions in this section.
Make an equation with eq, then name the variables to solve for. Other variables remain symbolic.</p>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Write</th><th>Meaning</th></tr>
<tr><td><code>eq(lhs,rhs)</code></td><td>An equation between the left and right expressions. It does not assign a value.</td></tr>
<tr><td><code>solve(equation,x)</code><br><code>solve([equation1,equation2],[x,y])</code></td><td>Solve the supplied equations for the distinct real variables you name.</td></tr>
<tr><td><code>steady_state(X)</code><br><code>steady_state([X,Y])</code></td><td>Set the rates for the named species to zero and solve for those species.</td></tr>
<tr><td><code>steady_state([X,Y],[equation1,equation2])</code></td><td>Add equations to the zero-rate equations. The extra equations do not add new unknowns.</td></tr>
<tr><td><code>stationary_state()</code><br><code>stationary_state([equation1,equation2])</code></td><td>Set every species rate to zero and solve for all species. The second form includes extra equations. Add any conservation totals you need explicitly.</td></tr>
<tr><td><code>subs(F,solutions)</code></td><td>Apply the relationships returned by solve, steady_state, or stationary_state to F.</td></tr>
</table>
<p>First, find the product concentration that gives a signal of 1 when gain is 2:</p>
<pre>solve(eq(subs(signal,gain,2),1),P)</pre>
<p>The answer is <code>P=1/4</code>. This query substitutes into the signal expression before
solving. A solve answer describes a relationship. To use it in another expression, apply it with
subs. A calculated number or expression alone is not a set of replacement relationships.</p>

<h3>Solve for Both Intermediates Together</h3>
<p>Enter <code>steady_state([X,Y])</code> to solve both zero-rate equations while keeping the rate
constants symbolic. Calculator displays the relationships it finds.</p>
<p>For a calculation with the five rate constants specified, substitute into the equations before
solving:</p>
<pre>solve(subs([eq(dX/dt,0),eq(dY/dt,0)],[kf1,kr1,kf2,kr2,k3],[2,3,5,7,11]),[X,Y])</pre>
<p>This gives <code>X=36*A/109, Y=10*A/109</code>. Apply those relationships to the product rate:</p>
<pre>subs(dP/dt,solve(subs([eq(dX/dt,0),eq(dY/dt,0)],[kf1,kr1,kf2,kr2,k3],[2,3,5,7,11]),[X,Y]))</pre>
<p>The rate is <code>10*A*k3/109</code>. The substitutions inside solve applied to the two equations,
not to the separate expression dP/dt. Insert k3=11 into that rate as well:</p>
<pre>subs(subs(dP/dt,solve(subs([eq(dX/dt,0),eq(dY/dt,0)],[kf1,kr1,kf2,kr2,k3],[2,3,5,7,11]),[X,Y])),k3,11)</pre>
<p>The result is <code>110*A/109</code>. These are complete queries, so you can enter each one
without saving the previous answer under a name.</p>
<p>Substituting solve answers does not change the rates used by dt. Differentiating a reduced
expression can bring back a species that was eliminated. Apply the relationships again after
each derivative step if you want to eliminate that species again. Solving zero-rate equations
does not prove stability or produce concentration trajectories.</p>

<h3>Substitute into Answers or Solve with New Values?</h3>
<p>To find solutions for particular parameter values, substitute into the equations before solving.
To evaluate answers already obtained, substitute into the solve result.
For the signal definition, setting gain to zero demonstrates the difference:</p>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Query</th><th>What Happens</th></tr>
<tr><td><code>subs(solve(eq(signal,0),P),gain,0)</code></td><td>Substitutes into the existing answer. P=0 remains. Substitution does not solve again.</td></tr>
<tr><td><code>solve(subs(eq(signal,0),gain,0),P)</code></td><td>The equation becomes 0=0 before solving, so P is free to take any value.</td></tr>
</table>
<p>An answer solved with symbols holds for general values of the other symbols, not necessarily
for a special value such as a zero gain or a zero rate constant. To solve for particular values,
use the second order of operations.
In solve answers, substitution changes the expressions on the right of =, not the variable
names on the left. The same variable name still means the same variable on both sides.</p>

<h2>Multiple Answers After Adding a Reaction</h2>
<p>Add <code>2 X -&gt; D; k=1</code> to <b>Reactions</b>. The complete example is now:</p>
<pre>A &lt;-&gt; X; kf=2; kr=3
X &lt;-&gt; Y; kf=5; kr=7
Y -&gt; P; k=11
2 X -&gt; D; k=1
param gain=2
param response=2*gain
let signal=response*[P]</pre>
<p>The added reaction uses the name <code>k4</code>. Use this four-reaction model for the following queries.</p>
<p>Find the concentrations of X at which the rate of dimerization equals the rate of conversion
to Y:</p>
<pre>solve(eq(k4*X^2,kf2*X),X)</pre>
<p>The two answers are <code>X=0</code> and <code>X=kf2/k4</code>.
Use <code>solution(result,n)</code> to choose usable answer n, starting at 1.
Select the second answer with:</p>
<pre>solution(solve(eq(k4*X^2,kf2*X),X),2)</pre>
<p>The selected relationship is <code>X=kf2/k4</code>. In the interface, choose that answer
from the selector and click <b>Use in Query</b> to insert its complete query.
Selection does not change which answers are included by <b>Copy Result</b>.</p>
<p>To calculate the rate for D for every answer, apply the unselected solve result to dD/dt:</p>
<pre>subs(dD/dt,solve(eq(k4*X^2,kf2*X),X))</pre>
<p>The two rate expressions are <code>0</code> and <code>kf2^2/k4</code>. To insert kf2=5 and
k4=1 in both expressions, enter:</p>
<pre>subs(subs(dD/dt,solve(eq(k4*X^2,kf2*X),X)),[kf2,k4],[5,1])</pre>
<p>The rate expressions become <code>0</code> and <code>25</code>.
For only the second answer, select the relationship before applying it:</p>
<pre>subs(dD/dt,solution(solve(eq(k4*X^2,kf2*X),X),2))</pre>
<p>Concentrations are nonnegative, so an answer that Calculator can show is negative is not listed. For example,
<code>solve(subs(eq(dX/dt,0),[A,Y,kf1,kr1,kf2,kr2,k4],[1,0,2,3,5,7,1]),X)</code> solves
<code>2-8*X-2*X^2=0</code> and gives only <code>X=-2+sqrt(5)</code>; the other root is negative.
This calculation sets only the rate for X to zero at the chosen A and Y.
It is not a steady state of the whole mechanism.</p>
<p>The examples from here onward use this four-reaction model with the signal definitions.</p>

<h3>Combine Results with Several Answers</h3>
<p>An ordinary expression, such as dD/dt above, can use every answer from one solve result.
A result with a single answer is an ordinary value and combines with anything.
When both operands contain several answers, select an answer from one operand first.
Calculator does not pair answers by their numbers or calculate every possible combination.</p>

<h2>Tuples, Matrices, and the Jacobian</h2>
<p>Square brackets group expressions into a tuple. For example, <code>[dP/dt,dD/dt]</code>
returns both product rates together. A single species in brackets, such as <code>[X]</code>,
still means a concentration. Write <code>[(X)]</code> for a one-entry tuple and
<code>[[(X)]]</code> for a nested tuple.</p>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Write</th><th>Meaning</th></tr>
<tr><td><code>odes()</code></td><td>A column matrix of all species rates. The displayed Rows line names their order.</td></tr>
<tr><td><code>jacobian()</code></td><td>A matrix of partial derivatives of the species rates. Rows identify rates and columns identify concentration variables, both in the species order used by odes().</td></tr>
<tr><td><code>definitions()</code></td><td>The definitions of the rate constants and equilibrium constants that are symbols of their own, as relationships for subs(F, definitions()).</td></tr>
<tr><td><code>matrix([[a,b],[c,d]])</code></td><td>Create a matrix from scalar entries. Replace a, b, c, and d with expressions. Rows must be nonempty and have the same length.</td></tr>
<tr><td><code>at(M,i)</code><br><code>at(M,i,j)</code></td><td>Extract item i from a tuple or column, or the matrix entry at row i and column j. Indices start at 0.</td></tr>
<tr><td><code>transpose(M)</code></td><td>Exchange rows and columns.</td></tr>
<tr><td><code>det(M)</code></td><td>Calculate the determinant of a square matrix.</td></tr>
</table>
<p>For the four-reaction example, odes() lists the species in the order A, D, P, X, Y.
The row for P is therefore 2 and the column for Y is 4:</p>
<pre>at(jacobian(),2,4)</pre>
<p>This gives <code>k3</code>, matching <code>partial(dP/dt,Y)</code>.
Likewise, <code>at(jacobian(),1,3)</code> gives <code>2*k4*X</code>, the derivative of the rate for
D with respect to X. Check the displayed species order when using indices with another model.
Matrix indices start at 0, whereas solution numbers start at 1.</p>
<p><code>eq(odes(),0)</code> makes a zero-rate equation for every matrix entry.
More generally, <code>eq(M,0)</code> compares each entry with zero, and
<code>eq(M,N)</code> compares corresponding entries of two matrices with the same shape.
These operations create equations. They do not choose unknowns or add conservation totals.</p>

<h2>Integrals, Limits, and Series</h2>
<p>These operations act on mathematical expressions. They do not solve the coupled kinetic
equations or produce concentration trajectories. The exponential expressions below are supplied
directly to demonstrate calculus with the parameter k3 from the example model.</p>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Write</th><th>Meaning</th></tr>
<tr><td><code>integrate(F,x)</code><br><code>integrate(F,x,a,b)</code></td><td>Find an antiderivative, or integrate from a to b. The indefinite form does not add an arbitrary integration constant. Symbolic bounds, unevaluated integrals, and signed infinity can appear in results.</td></tr>
<tr><td><code>limit(F,x,a)</code><br><code>limit(F,x,a,left)</code><br><code>limit(F,x,a,right)</code></td><td>Find a limit at a. The default is two-sided at a finite point. Use the literal, unquoted word left or right for a one-sided limit. At infinity, the direction follows the endpoint.</td></tr>
<tr><td><code>series(F,x,a,n)</code><br><code>series(F,x,a,n,left)</code><br><code>series(F,x,a,n,right)</code></td><td>Expand around a to positive integer order n. The default direction is right. The result retains an order term indicating the truncation.</td></tr>
<tr><td><code>remove_order(F)</code></td><td>Remove the top-level series order term before substituting for the expansion variable. When an order term is removed, a truncation notice remains.</td></tr>
</table>
<p>Start by inserting k3=11 into an exponential, then integrate with respect to t:</p>
<pre>integrate(subs(exp(-k3*t),k3,11),t)</pre>
<p>An antiderivative is <code>-exp(-11*t)/11</code>.
For a definite integral from t=0 to t=1, enter:</p>
<pre>integrate(subs(exp(-k3*t),k3,11),t,0,1)</pre>
<p>The result is <code>(1-exp(-11))/11</code>.
A limit can keep k3 symbolic. For example,
<code>limit((1-exp(-k3*t))/t,t,0)</code> gives <code>k3</code>.</p>

<h3>An Integral to Infinity</h3>
<pre>integrate(exp(-k3*t),t,0,infinity())</pre>
<p>The integral is <code>1/k3</code>, because rate constants are positive. To apply k3=11 to
this result, enter:</p>
<pre>subs(integrate(exp(-k3*t),t,0,infinity()),k3,11)</pre>
<p>The result is <code>1/11</code>. A zero rate constant is a special value: substitute it before
integrating. <code>integrate(subs(exp(-k3*t),k3,0),t,0,10)</code> gives <code>10</code>, as does
<code>limit(integrate(exp(-k3*t),t,0,10),k3,0)</code>, while substituting k3=0 into the
integrated formula is undefined.</p>

<h3>Evaluate a Truncated Series</h3>
<p>Expand the same exponential around t=0:</p>
<pre>series(exp(-k3*t),t,0,3)</pre>
<p>The series is <code>1-k3*t+k3^2*t^2/2+O(t^3)</code>.
Order 3 keeps powers below t cubed in this example. Remove the order term to work with just the
polynomial:</p>
<pre>remove_order(series(exp(-k3*t),t,0,3))</pre>
<p>Then evaluate that polynomial at t=1/100 and k3=11:</p>
<pre>subs(remove_order(series(exp(-k3*t),t,0,3)),[t,k3],[1/100,11])</pre>
<p>This gives <code>17921/20000</code> with a truncation notice.
It is the value of the truncated polynomial, not the exact value of <code>exp(-11/100)</code>.
Use the complete query to reuse a series result. The displayed O notation is not an input command.</p>

<h2>Scalar Functions</h2>
<p>Replace the arguments below with expressions from your model or literal values.
Write the parentheses in <code>pi()</code>, <code>e()</code>, and <code>infinity()</code>.</p>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Write</th><th>Meaning</th></tr>
<tr><td><code>pi()</code>, <code>e()</code>, <code>infinity()</code></td><td>Exact pi, Euler's number, and positive infinity. Use <code>-infinity()</code> for negative infinity.</td></tr>
<tr><td><code>sqrt(x)</code>, <code>pow(x,y)</code></td><td>Principal square root and x raised to y. Complex results are allowed.</td></tr>
<tr><td><code>exp(x)</code></td><td>The exponential function, e raised to x.</td></tr>
<tr><td><code>expm1(x)</code></td><td><code>exp(x)-1</code>.</td></tr>
<tr><td><code>ln(x)</code>, <code>log(x)</code></td><td>Natural logarithm. Both forms take one argument.</td></tr>
<tr><td><code>log10(x)</code></td><td>Base-ten logarithm.</td></tr>
<tr><td><code>log1p(x)</code></td><td>Natural logarithm of 1+x.</td></tr>
<tr><td><code>sin(x)</code>, <code>cos(x)</code>, <code>tan(x)</code></td><td>Trigonometric functions with angles in radians.</td></tr>
<tr><td><code>abs(x)</code>, <code>min(x,...)</code>, <code>max(x,...)</code></td><td>Absolute value and the minimum or maximum of one or more real expressions.</td></tr>
<tr><td><code>erf(x)</code>, <code>heaviside(x)</code></td><td>Error function and step function. The step is 0 for x&lt;=0 and 1 for x&gt;0.</td></tr>
<tr><td><code>clip(x,lo,hi)</code></td><td>Clamp x between lo and hi. If lo is greater than hi, the bounds are swapped.</td></tr>
</table>

<h2>Answers, Unavailable Results, and Limitations</h2>
<p>A result can contain usable answers and unavailable results together:</p>
<ul>
<li><b>Usable answers</b> provide expressions or relationships you can use in further calculations.
Only usable answers receive selectable solution numbers.</li>
<li><b>Unavailable results</b> include an explanation of why a calculation could not be used.
They cannot be selected.</li>
</ul>
<p>Answers hold for general values of the variables in their domains. A special value, such as a
rate constant of zero, can need its own calculation: substitute it before solving or integrating.
For a derivative at that value, take the limit, for example
<code>limit(partial(integrate(exp(-k3*t),t,0,10),k3),k3,0)</code>.
Substituting into a finished answer can be undefined, for example when the answer divides by that
rate constant.</p>
<p>Calculator uses SymPy for the mathematics and shows what SymPy returns. SymPy can miss solutions,
list an answer whose sign it cannot decide, leave an equation, integral or limit unsolved, or compute
some integrals and limits wrongly.
Calculator finds every solution of linear equations, every solution SymPy reports for a single
equation in one variable unless SymPy leaves part of it unsolved, which Calculator reports as unsolved, and every solution of polynomial equations with rational coefficients whose solutions one requested
coordinate fixes. A single equation in exponentials of the variable whose exponents are rational
multiples of one another, such as <code>exp(-t)</code> and <code>exp(-3*t/2)</code>, is solved as a
polynomial in one exponential. For these equations, an empty result means SymPy found no solutions in the coordinate domains. With numerical
coefficients, roots of such polynomials beyond the quadratic are shown exactly as CRootOf; use approx
for decimals.
Other equations, including other polynomial systems, use SymPy's general solvers. Calculator shows only their answers that SymPy
confirms satisfy every equation, with the notice that the solver can miss solutions; when none
remain, Calculator reports that it could not find solutions.
An equation with infinitely many solutions, such as a periodic one, gives a solution set, which
cannot be applied with subs. So does an answer that holds only over a range of the variable, such as
one from max, min or clip of the variable with a numerical threshold; with a parameter threshold, such
as a concentration floor, the equation is unsolved, so substitute values for the threshold and for
any other concentrations inside the max first.
A max, min or other case formula of other quantities stays a coefficient of the answer. A case
that holds only at a single value of another quantity, such as k1=k2 in an integral result, is set
aside, as for any general answer; an equation whose cases depend on other quantities over a range is
unsupported until values are substituted for them.</p>
<p>Answers that Calculator can show are negative for a concentration, taking concentrations as
nonzero, zero or negative for a positive quantity such as T or a rate constant, or not real for a
real variable, are not listed. An answer whose sign SymPy cannot decide is listed.
Substitution inserts any value: <code>subs(X,X,sqrt(-1))</code> gives <code>sqrt(-1)</code>.</p>

<h3>Reuse the Calculation When the Displayed Expression Is Not Valid Input</h3>
<p>Some outputs contain functions or notation that Calculator can display but cannot accept
directly as input. For example, gamma may appear in an integral result, but it is not an input
command. Solution sets, some constants, roots, order terms, and numbers outside the accepted
input range can also require the complete query.
Use <b>Use Result</b>, or select an answer and use <b>Use in Query</b>, to insert that query.
It includes the solves and substitutions needed to reproduce the answer.</p>
<p>If a definition cannot be calculated, you can still evaluate queries that do not use it.
Queries that need the failed definition show that failure.</p>

<h3>Substitution and Calculus</h3>
<p>Substitution keeps the variable of an unevaluated integral separate from a replacement that
uses the same name; the integration variable is then shown as _0, and an antiderivative is
written as its integral up to that variable, such as <code>Integral(f(_0), (_0, A))</code>.
Results that SymPy writes with cases, such as an integral whose value depends on the sign of a
parameter, keep those cases in later calculations. Derivatives, limits and series use the formula
for general values where a case holds only at single values of their variable, such as gain=0.
A limit of a formula whose cases depend on its variable over a range, such as
<code>limit(integrate(exp(-gain*t),t,0,infinity()),gain,1)</code>, is unsupported. Integrating an
earlier integral result over a rate constant from 0, such as
<code>integrate(integrate(exp(-k3*t),t,0,10),k3,0,1)</code>, can give a spurious imaginary part.
Symbolic calculus does not establish where the original expression is defined or prove
smoothness.</p>

<h3>Integrals of Rational Functions and Unsupported Models</h3>
<p>SymPy 1.14.0 can lose terms when it writes the integral of a rational term in real form and
the term has parameters and a denominator with a quadratic or higher factor
(<code>sympy/sympy#28596</code>). For such a term, such as <code>integrate(1/(X^2-gain),X)</code>,
Calculator gives the antiderivative with complex logarithms. A definite integral with such a
term, such as <code>integrate(1/(X^2-gain),X,0,1)</code>, is unsupported; substitute values for
the parameters first, or apply approx to the integrand when physical constants such as R remain.</p>
<p>Calculator does not support conditions that change during a run, logical observables,
fractional mechanisms or schedules. Solving equations also does not establish that the answers
are physically valid, reachable from the starting concentrations or suitable as a reduced kinetic
model.</p>
"""
