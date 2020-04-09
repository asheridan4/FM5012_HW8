import numpy as np 
import scipy.stats as si
import sympy as sym     
import scipy.optimize as opt 

# The following is the code I used for #3.

def TwoVar_Newton(function, X, Y, maxiter):
    f = function
    loopcount = 0
    delta = 0
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    xval = X
    yval = Y
    while loopcount < maxiter:
        val_prime_x = sym.diff(f, x).evalf(subs={x: xval, y: yval})
        val_prime_y = sym.diff(f, y).evalf(subs={x: xval, y: yval})
        gradient = np.array([val_prime_x, val_prime_y])
        val_double_xx = sym.diff(sym.diff(f, x), x).evalf(subs={x: xval, y: yval}) + 0.0
        val_double_xy = sym.diff(sym.diff(f, x), y).evalf(subs={x: xval, y: yval}) + 0.0
        val_double_yx = sym.diff(sym.diff(f, y), x).evalf(subs={x: xval, y: yval}) + 0.0
        val_double_yy = sym.diff(sym.diff(f, y), y).evalf(subs={x: xval, y: yval}) + 0.0
        determinant = (val_double_xx * val_double_yy) - (val_double_xy * val_double_yx)
        hess_inv = np.matrix([[val_double_yy / determinant, -1 * val_double_xy / determinant], [-1 * val_double_yx / determinant, val_double_xx / determinant]]) 
        delta = np.dot(hess_inv, gradient)
        xval = xval - delta[0, 0]
        yval = yval - delta[0, 1]
        loopcount += 1
        print("The estimate of the stationary point is", np.array([xval, yval]) , "after", loopcount, "iterations")



print("The solutions to #3 begin here")
x = sym.Symbol('x')
y = sym.Symbol('y')
testfunc = 100 * ((y-x**2)**2)+ ((1-x)**2)
print(TwoVar_Newton(testfunc, 4, 4, 7))
print("The solutions to #3 end here")
print("")

# The following is the code I used for #4.
def Gen_Bisection(function, min, max):
    f = function
    lbound = min
    rbound = max
    r = sym.Symbol('r')
    lboundval = sym.diff(f, r).evalf(subs={r: lbound})
    rboundval = sym.diff(f, r).evalf(subs={r: rbound})
    loopcount = 0
    width = rbound - lbound
    while width > .01:
        if lboundval * rboundval >= 0:
            print("Bisection Method Fails")
            width = 0
        else:
            newbound = (lbound + rbound) / 2
            newboundval = sym.diff(f, r).evalf(subs={r: newbound})
            if lboundval * newboundval < 0:
                rbound = newbound
                rboundval = newboundval
                width = rbound - lbound
                loopcount += 1
            else:
                lbound = newbound
                lboundval = newboundval
                width = rbound - lbound
                loopcount += 1
    else:
        if width == 0:
            pass
        else:
            xans = (lbound + rbound) /2
            yans = f.evalf(subs={r: xans})
            ans = np.array([xans, yans])
            print("The stationary point", ans, "was found after", loopcount, "iterations using Bisection Method")
            
def Gen_Newton(function, x0):
    f = function
    r = sym.Symbol('r')
    loopcount = 0
    xn = x0
    deltaxn = 1
    while np.absolute(deltaxn) > .01:
        deltaxn = -1 * sym.diff(f, r).evalf(subs={r: xn}) / sym.diff(sym.diff(f, r), r).evalf(subs={r: xn})
        xn = xn + deltaxn
        loopcount += 1
        #print("The x value of the new point is", xn, "after", loopcount, "iterations")
    else:
        ans = np.array([xn, f.evalf(subs={r: xn})])
        print("The stationary point", ans, "was found after", loopcount, "iterations using Newton's Method")

def Gen_Secant(function, min, max, maxiter):
    f = function
    lbound = min
    rbound = max
    r = sym.Symbol('r')
    lboundval = sym.diff(f, r).evalf(subs={r: lbound})
    rboundval = sym.diff(f, r).evalf(subs={r: rbound})
    loopcount = 0
    width = rbound - lbound
    while loopcount < maxiter: 
        if lboundval * rboundval >= 0:
            print("Secant Method Fails")
            loopcount = maxiter
            width = 0
        else:
            newbound = lbound - lboundval * (width / (rboundval - lboundval))
            newboundval = sym.diff(f, r).evalf(subs={r: newbound})
            if lboundval * newboundval < 0:
                rbound = newbound
                rboundval = newboundval
                width = rbound - lbound
                loopcount += 1
                #print("The x value of the new point is", newbound, "after", loopcount, "iterations")
            else:
                lbound = newbound
                lboundval = newboundval
                width = rbound - lbound
                loopcount += 1
                #print("The x value of the new point is", newbound, "after", loopcount, "iterations")
    else:
        if width == 0:
            pass
        else:
            xans = newbound
            yans = f.evalf(subs={r: xans})
            ans = np.array([xans, yans])
            print("The stationary point", ans, "was found after", loopcount, "iterations using the Secant Method")
r = sym.Symbol('r')
fourthfunc = 10/(1+r) + 10/((1+r)**2) + 10/((1+r)**3) + 110/((1+r)**4)
print("The solutions to #4 begin here")
print("This is the solution using Bisection:")
Gen_Bisection(fourthfunc, -6, -4)
print("This is the solution using Secant Method:")
Gen_Secant(fourthfunc, -6, -4, 15)
print("This is the solution using Newton's Method:")
Gen_Newton(fourthfunc, -3)
print("The soultions to #4 end here")
print("")

# The following is the code used for #5. Not all code shown below was used, but it was copied over from an assignment for 5091.
d1_val = lambda S,K,T,r,v: (np.log(S / K) + (r + .5 * v ** 2) * T) / (v * np.sqrt(T))
d2_val = lambda S,K,T,r,v: (np.log(S / K) + (r - .5 * v ** 2) * T) / (v * np.sqrt(T))
call_value = lambda S,K,T,r,v: (S * si.norm.cdf(d1_val(S,K,T,r,v), 0, 1)) - (K * np.exp(-r * T) * si.norm.cdf(d2_val(S,K,T,r,v), 0, 1))
put_value = lambda S,K,T,r,v: (K * np.exp(-r * T) * si.norm.cdf(-1 * d2_val(S,K,T,r,v), 0, 1)) - (S * si.norm.cdf(-1 * d1_val(S,K,T,r,v), 0, 1))
call_delta = lambda S,K,T,r,v: si.norm.cdf(d1_val(S,K,T,r,v), 0, 1)
put_delta = lambda S,K,T,r,v: -si.norm.cdf(-1 * d1_val(S,K,T,r,v), 0, 1)
gamma_value = lambda S,K,T,r,v: (si.norm.pdf(d1_val(S,K,T,r,v), 0, 1)) / (S * v * np.sqrt(T))
vega_value = lambda S,K,T,r,v: (S * np.sqrt(T) * si.norm.pdf(d1_val(S,K,T,r,v), 0, 1))
call_theta = lambda S,K,T,r,v: (((-1 * S * si.norm.pdf(d1_val(S,K,T,r,v), 0, 1) * v) / (2 * np.sqrt(T))) - (r * K * np.exp(-r * T) * si.norm.cdf(d2_val(S,K,T,r,v), 0, 1)))
put_theta = lambda S,K,T,r,v: (((-1 * S * si.norm.pdf(d1_val(S,K,T,r,v), 0, 1) * v) / (2 * np.sqrt(T))) + (r * K * np.exp(-r * T) * si.norm.cdf(-1 * d2_val(S,K,T,r,v), 0, 1)))
call_rho = lambda S,K,T,r,v: (K * T * np.exp(-r * T) * si.norm.cdf(d2_val(S,K,T,r,v), 0, 1))
put_rho = lambda S,K,T,r,v: (-K * T * np.exp(-r * T) * si.norm.cdf(-1 * d2_val(S,K,T,r,v), 0, 1))


def Call_Bisection(S,K,T,r,C):
    upbound = 10
    lowbound = .001
    rboundval = call_value(S,K,T,r,upbound)
    lboundval = call_value(S,K,T,r,lowbound)
    loopcount = 0
    width = upbound - lowbound

    while width > .01:           # <-- adjust tolerance for "Call_Bisection" here
        if max(rboundval, lboundval) > C and min(rboundval,lboundval) < C:
            newbound = (upbound + lowbound) / 2
            nboundval = call_value(S,K,T,r,newbound)
            if nboundval > C:
                if rboundval > lboundval:
                    upbound = newbound
                    width = upbound - lowbound
                    loopcount += 1
                else:
                    lowbound = newbound
                    width = upbound - lowbound
                    loopcount += 1
            else:
                if rboundval > lboundval:
                    lowbound = newbound
                    width = upbound - lowbound
                    loopcount += 1
                else:
                    upbound = newbound
                    width = upbound - lowbound
                    loopcount += 1
        else:
            print("Unbounded: Implied Volatility is 0%", "or greater than 1000%")
            width = 0
    else:
        if width == 0:
            pass
        else:
            ans = (upbound + lowbound) /2
            return(np.array([ans, loopcount]))



def Call_Newton(S,K,T,r,C):
    xn = 1
    deltaxn = call_value(S,K,T,r,xn)  - C
    loopcount = 0

    while np.absolute(deltaxn) > .01:            # <-- Adjust tolerance for "Call_Newton" here
        deltaxn = ((C - call_value(S,K,T,r,xn)) / vega_value(S,K,T,r,xn))
        xn = xn - (call_value(S,K,T,r,xn) - C) / vega_value(S,K,T,r,xn)
        loopcount += 1
    else:
        return(np.array([xn, loopcount]))

def Call_Secant(S,K,T,r,C):
    upbound = 10
    lowbound = .001
    rboundval = call_value(S,K,T,r,upbound)
    lboundval = call_value(S,K,T,r,lowbound)
    loopcount = 0
    width = upbound - lowbound
    xn = 1
    f_a = C - lboundval
    f_b = C - rboundval

    while width > .01:           # <-- adjust tolerance for "Call_Bisection" here
        if f_a * f_b >= 0:
            print("Secant Method Fails")
            width = 0
        else:
            xn = lowbound - f_a * (width / (f_b - f_a))
            newboundval = call_value(S, K, T, r, xn)
            f_new = C - newboundval
            if f_a * f_new < 0:
                upbound = xn
                rboundval = newboundval
                width = upbound - lowbound
                loopcount += 1
            else:
                lowbound = xn
                lboundval = newboundval
                width = upbound - lowbound
                loopcount += 1
    else:
        if width == 0:
            pass
        else:
            ans = (upbound + lowbound) /2
            return(np.array([ans, loopcount]))

print("The solutions to #5 begin here:")
print("These are tests of the Bisection methods with 4 different C values")
print(Call_Bisection(100, 100, .5, .01, 10))
print(Call_Bisection(100, 100, .5, .01, 11))
print(Call_Bisection(100, 100, .5, .01, 12))
print(Call_Bisection(100, 100, .5, .01, 13))
print("These are tests of Newton's Method with 4 different C values")
print(Call_Newton(100, 100, .5, .01, 10))
print(Call_Newton(100, 100, .5, .01, 11))
print(Call_Newton(100, 100, .5, .01, 12))
print(Call_Newton(100, 100, .5, .01, 13))
print("These are tests of the Secant Method with 4 different C values")
print(Call_Secant(100, 100, .5, .01, 10))
print(Call_Secant(100, 100, .5, .01, 11))
print(Call_Secant(100, 100, .5, .01, 12))
print(Call_Secant(100, 100, .5, .01, 13))
print("The solutions to #5 end here")
