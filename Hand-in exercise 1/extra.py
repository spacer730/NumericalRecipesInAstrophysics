def exttrapezoid(func, edges, n):
   h = (edges[1]-edges[0])/n
   integration = 0.5*(func(edges[0])+func(edges[1]))

   for i in range(1,n):
      integration += func(edges[0]+h*i)
   integration = h*integration

   return integration

   """
   more efficient way for doing extended-midpoint-romberg if I know how to later on. Not sure how to get the new points efficiently
   for i in range(1,N):
      newint = 0.5*(s[0,i])+h*(newpoints)
      s[0].append(newint)
   """
