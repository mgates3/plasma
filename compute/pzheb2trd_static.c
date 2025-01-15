/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "bulge.h"
#include "core_blas.h"
#include <omp.h>
#include <sched.h>
#include <string.h>

#undef REAL
#define COMPLEX

/***************************************************************************//**
 *  Static scheduler
 **/

#define shift 3

#define ss_cond_set(m, n, val)                  \
    {                                                   \
        plasma->ss_progress[(m)+plasma->ss_ld*(n)] = (val); \
    }


#define ss_cond_wait(m, n, val) \
    {                                                           \
        while (plasma->ss_progress[(m)+plasma->ss_ld*(n)] != (val)) \
            sched_yield();                                          \
    }


//  Parallel bulge chasing column-wise - static scheduling

void plasma_pzheb2trd_static( plasma_enum_t uplo, int N, int NB, int Vblksiz,
			 plasma_complex64_t *A, int LDA,
			 plasma_complex64_t *V, plasma_complex64_t *TAU,
			 double *D, double *E, int WANTZ,
			 plasma_workspace_t work,
			 plasma_sequence_t *sequence, plasma_request_t *request) 
{

    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return;
    }    
    
    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    if ( uplo != PlasmaLower ) {
        plasma_request_fail(sequence, request, PlasmaErrorNotSupported);
        return;
    }
    
    
    // Quick return
    if (N == 0) {
        return;
    }

    /*
     * General case:
     *
     * As I store V in the V vector there are overlap between
     * tasks so shift is now 4 where group need to be always
     * multiple of 2 (or shift=5 if not multiple of 2),
     * because as example if grs=1 task 2 from
     * sweep 2 can run with task 6 sweep 1., but task 2 sweep 2
     * will overwrite the V of tasks 5 sweep 1 which are used by
     * task 6, so keep in mind that group need to be multiple of 2,
     * and thus tasks 2 sweep 2 will never run with task 6 sweep 1.
     * OR if we allocate V as V(N,2) and we switch between the storing of
     * sweep's like odd in V(N,1) and even in V(N,2) then no overlap and so
     * shift is 3.
     * when storing V in matrix style, shift could be back to 3.
     * */
    
    /* Some tunning for the bulge chasing code
     * see technical report for details */
    int nbtiles = plasma_ceildiv(N,NB);
    int colblktile = 1;
    int grsiz = 1;    
    int maxrequiredcores = imax( nbtiles/colblktile, 1 );
    int colpercore = colblktile*NB;
    int thgrsiz = N;
    
    
    // Initialize static scheduler progress table
    int cores_num;
    #pragma omp parallel 
    {
        cores_num  = omp_get_num_threads();
    }
    int size = 2*nbtiles+shift+cores_num+10;
    plasma->ss_progress = (volatile int *)malloc(size*sizeof(int));
    for(int index = 0; index < size; index++) plasma->ss_progress[index] = 0;
    plasma->ss_ld = (size);
    
    // main bulge chasing code 
    int ii = shift/grsiz;
    int  stepercol =  ii*grsiz == shift ? ii:ii+1;
    ii       = (N-1)/thgrsiz;
    int thgrnb  = ii*thgrsiz == (N-1) ? ii:ii+1;
    int allcoresnb = imin( cores_num, maxrequiredcores );

    #pragma omp parallel
    {   
        int coreid, sweepid, myid, stt, st, ed, stind, edind;
        int blklastind, colpt,  thgrid, thed;
        int i,j,m,k;

        int my_core_id = omp_get_thread_num();
        plasma_complex64_t  *WORK = work.spaces[my_core_id];

        for (thgrid = 1; thgrid<=thgrnb; thgrid++){
            stt  = (thgrid-1)*thgrsiz+1;
            thed = imin( (stt + thgrsiz -1), (N-1));
            for (i = stt; i <= N-1; i++){
                ed = imin(i,thed);
                if(stt>ed) break;
                for (m = 1; m <=stepercol; m++){
                    st=stt;
                    for (sweepid = st; sweepid <=ed; sweepid++){
                        
                        for (k = 1; k <=grsiz; k++){
                            myid = (i-sweepid)*(stepercol*grsiz) +(m-1)*grsiz + k;
                            if(myid%2 ==0){
                                colpt      = (myid/2)*NB+1+sweepid-1;
                                stind      = colpt-NB+1;
                                edind      = imin(colpt,N);
                                blklastind = colpt;
                            } else {
                                colpt      = ((myid+1)/2)*NB + 1 +sweepid -1 ;
                                stind      = colpt-NB+1;
                                edind      = imin(colpt,N);
                                if( (stind>=edind-1) && (edind==N) )
                                    blklastind=N;
                                else
                                    blklastind=0;
                            }
                            coreid = (stind/colpercore)%allcoresnb;
                            
                            if(my_core_id==coreid) {
                                if(myid==1) {
                                    
                                    ss_cond_wait(myid+shift-1, 0, sweepid-1);
                                    core_zhbtype1cb(N, NB, A, LDA, V, TAU, stind-1, edind-1, sweepid-1, Vblksiz, WANTZ, WORK);
                                    ss_cond_set(myid, 0, sweepid);
                                    
                                    if(blklastind >= (N-1)) {
                                        for (j = 1; j <= shift; j++)
                                            ss_cond_set(myid+j, 0, sweepid);
                                    }
                                } else {
                                    ss_cond_wait(myid-1,       0, sweepid);
                                    ss_cond_wait(myid+shift-1, 0, sweepid-1);
                                    if(myid%2 == 0)
                                        core_zhbtype2cb(N, NB, A, LDA, V, TAU, stind-1, edind-1, sweepid-1, Vblksiz, WANTZ, WORK);
                                    else
                                        core_zhbtype3cb(N, NB, A, LDA, V, TAU, stind-1, edind-1, sweepid-1, Vblksiz, WANTZ, WORK);
                                    
                                    ss_cond_set(myid, 0, sweepid);
                                    if(blklastind >= (N-1)) {
                                        for (j = 1; j <= shift+allcoresnb; j++)
                                            ss_cond_set(myid+j, 0, sweepid);
                                    }
                                } /* END if myid==1 */
                            } /* END if my_core_id==coreid */
                            
                            if(blklastind >= (N-1)) {
                                stt++;
                                break;
                            }
                        } /* END for k=1:grsiz */
                    } /* END for sweepid=st:ed */
                } /* END for m=1:stepercol */
            } /* END for i=1:N-1 */
         } /* END for thgrid=1:thgrnb */
    }
    /* finalize static sched */
    free((void*)plasma->ss_progress);
    
    /*================================================
     *  store resulting diag and lower diag D and E
     *  note that D and E are always real
     *================================================*/
    /*
     * STORE THE RESULTING diagonal/off-diagonal in D AND E
     */
    /* Make diagonal and superdiagonal elements real,
     * storing them in D and E
     */
    /* In complex case, the off diagonal element are
     * not necessary real. we have to make off-diagonal
     * elements real and copy them to E.
     * When using HouseHolder elimination,
     * the ZLARFG give us a real as output so, all the
     * diagonal/off-diagonal element except the last one are already
     * real and thus we need only to take the abs of the last
     * one.
     *  */
    // sequential code here so only core 0 will work 
    if( uplo == PlasmaLower ) {
        for (int i=0; i < N-1 ; i++) {
            D[i] = creal(A[i*LDA]);
            E[i] = creal(A[i*LDA+1]);
        }
        D[N-1] = creal(A[(N-1)*LDA]);
    } else { /* PlasmaUpper not tested yet */
        for (int i=0; i<N-1; i++) {
            D[i] = creal(A[i*LDA+NB]);
            E[i] = creal(A[i*LDA+NB-1]);
        }
        D[N-1] = creal(A[(N-1)*LDA+NB]);
    } /* end PlasmaUpper */
    
    return;
}
