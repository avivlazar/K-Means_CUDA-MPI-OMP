
///////////////////////////////////////////////
// MPI - Maset, Slaves, and Slaves' Listener //
///////////////////////////////////////////////

General idea:
	There are 3 types of actors: master, slaves and slaves' listener.
	Master and the slaves have one goal: to find the first picture* which: Q < QM
	Slaves' listener goal is to be the element which connect between the master and the slaves, and arange which of them calc the current picture (dt).
	Master and slaves' listener exist in the same machine, which is the master. They are threads (OMP)
	
	Master and slaves' listener' talk with each other by 'AllData' struct. They check each time if the other one found a good result.
	It's the resposibility of master and slaves' listener to stop their work when the someone (master or one of the slaves) find a good result.

	In the end of the calculation, the master decide the final result from two options:
	1) The result of the master itself
	2) The result of slaves.
	
	If there are two good results - the master chooses by the prefference: minimal picture ID - which means a lower period' time

	*Defenition of picture: picture A is the status of the points after A dt




/////////////////
// Calc Picture //
/////////////////

General Steps in calculation of a single picture:

Assampions:
	N <= 3,000,000 , 2<=K<=N


	1) calc the new locations of the current picture. 
	   It does by the picture' id and speed' equation:  
		old_location + (t*speed)
		// t = id*dt
		runtime complexity: O(N / (OMP + CUDA))
	2) init the clusters' centers to be the location of the point 
		which is in the same position in thier array.
		exampe: cluster 5' center will be the location of point number 5 in points array 
		the init is done by OMP
		runtime complexity: O(K)
	3) Find the real clusters' centers which answer the question:
		Is there optimal centers for K clusters to the N points in this picture? 
		runtime complexity: O(LIMIT*K*N / (OMP + CUDA))
	4) Calc the diameter of each cluster
	    runtime complexity: O((N^2 / K) / (OMP + CUDA)) -> in AVERAGE case
	5) calc and set the points' locations in time: t = 0
		with the same equation in step 2.
		runtime complexity: O(N / (OMP + CUDA))

	* WARNING: in the code there are more steps for promising the runtime complexity.
				There are more details about them inside the code.


More about finding clusters' center:

Enter to a loop which take in charge 3 conditions for continue it:
	- number of iterations it does doesn't over the LIMIT
	- the clusters' centers is located in special loctions where
		they can't be changed because the system is stable
		* the system is stable when all N points didn't change cluster' id' belonging
		the agenda behind it is so:
		if the LIMIT is high (for example: 100) and afters 30 iterations the system is stable,
		we can save 70 wasted iterations which do nothing.

Steps of each iteration (0 <= iteration < LIMIT):
	1) decide for each point which from the K clusters which its center
		is the closest one to the point location.
		each point "remember" her cluster index (0 to K-1)
		runtime complexity: O(N*K / (OMP + CUDA))
	2) Calc clusters' centers due to Step 1, and judge if the centers where moved. If so, the system is not stable yet. 
		runtime complexity: O(N / OMP)


More about finding diameter of each cluster:
		Our basic assumption was: (N >= K), which means that the number of points is larger (or equal) than
		the number of clusters. That's why we created an array of points' indices (int*) for each cluster
   
		In the best case: each one of the K clusters contains exectly N/K points' indices.
		From the proccess of finding cluster, we need to scan all the points' indices of each cluster.
		In serial proccess, the runtime comlexity is:
			-> O(K * (N/K) * (N/K)) = O(N^2/K)
		when in "naive" serial solution, e need to scan all N points, which means: O(N^2).
		Because of the basic assumption which says: K > 1, so: O(N^2/K) < O(N^2),
		when the algorithm for scanning and build the indices array have runtime comlexity of: O(N)
			Conclusion: O(N^2/K) + O(N) < O(N^2)

		However - in the worst case, the proccess' runtime comlexity will be O(N^2).

		In average case, the runtime comlexity is O(N^2\K)