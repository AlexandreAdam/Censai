C
C       ============================================
C       Purpose: Compute exponential integral Ei(x)
C       Input :  x  --- Argument of Ei(x)
C       Output:  EI --- Ei(x)
C       ============================================
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        IF (X.EQ.0.0) THEN
           EI=-1.0D+300
        ELSE IF (X .LT. 0) THEN
           CALL E1XB(-X, EI) ----- We are here
           EI = -EI
#         ELSE IF (DABS(X).LE.40.0) THEN
# C          Power series around x=0
#            EI=1.0D0
#            R=1.0D0
#            DO 15 K=1,100
#               R=R*K*X/(K+1.0D0)**2
#               EI=EI+R
#               IF (DABS(R/EI).LE.1.0D-15) GO TO 20
# 15         CONTINUE
# 20         GA=0.5772156649015328D0
#            EI=GA+DLOG(X)+X*EI
#         ELSE
# C          Asymptotic expansion (the series is not convergent)
#            EI=1.0D0
#            R=1.0D0
#            DO 25 K=1,20
#               R=R*K/X
# 25            EI=EI+R
#            EI=EXP(X)/X*EI
#         ENDIF
#         RETURN
#         END


        SUBROUTINE E1XB(X,E1)
C
C       ============================================
C       Purpose: Compute exponential integral E1(x)
C       Input :  x  --- Argument of E1(x)
C       Output:  E1 --- E1(x)  ( x > 0 )
C       ============================================
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        IF (X.EQ.0.0) THEN
           E1=1.0D+300
        ELSE IF (X.LE.1.0) THEN ---   case - < 1.
           E1=1.0D0
           R=1.0D0
           DO 10 K=1,25  -- 10 is a label, it is meaningless, K from 1 to 25
              R=-R*K*X/(K+1.0D0)**2
              E1=E1+R
              IF (DABS(R).LE.DABS(E1)*1.0D-15) GO TO 15
10         CONTINUE
15         GA=0.5772156649015328D0
           E1=-GA-DLOG(X)+X*E1    --- DLOG is just log, so this is -gamma - log(x) + x * E1
        ELSE
           M=20+INT(80.0/X)
           T0=0.0D0
           DO 20 K=M,1,-1
              T0=K/(1.0D0+K/(X+T0))
20         CONTINUE
           T=1.0D0/(X+T0)
           E1=EXP(-X)*T
        ENDIF
        RETURN
        END