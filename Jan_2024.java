////////////////////////////////////////////
// Vertex Cover
//  21JAN24
////////////////////////////////////////////
class Solution {
        public static int ans;
        public static ArrayList<int[]> edge;
        public static int vertexCover(int n, int m, int[][] edges) {
            // code here
            /*
            vertex cover in an undirected graph is the set of vertices, where every edge from a vertex not in the vertex cover
            shares every edge with a vertex in the vertic covr
            rather for every edge in the graph, at least on of the enpoind sohuld belong to the vertex cover
            there would be many possible solutions for the vertex cover set, we want the one which is smallest
            minimum vertex cover
            n is number of nodes and m is number of edges
            two big to try all possible sets, espeically since edges can be N**2
            can be solved with backtracking take, and add to set, then verify if we can touch all
            */
            edge = new ArrayList<>();
            ans = n;
            //put into new edge list
            for (int[] e : edges){
                edge.add(e);
            }
            ArrayList<Integer> vertex = new ArrayList<>(n); //if we've taken this vertex i
            for (int i = 0; i < n; i++){
                vertex.add(0); //not tataken
            }
            backtrack(0,vertex);
            return ans;
        
        }
        public static int count(ArrayList<Integer> vertex){
            //valiate if all edges touch the vertex cover
            for (int[] e : edge){
                if (vertex.get(e[0] - 1) == 0 && vertex.get(e[1] - 1) == 0){
                    return ans;
                }
            }
            int c = 0;
            for (int v : vertex){
                c += v;
            }
            return c;
            
        }
        public static void backtrack(int i, ArrayList<Integer> vertex){
            //got to the end
            if (i == vertex.size()){
                //we want the smallest one
                int val = count(vertex);
                ans = Math.min(val,ans);
                return;
            }
            backtrack(i+1,vertex);
            vertex.set(i,1);
            backtrack(i+1,vertex);
            vertex.set(i,0);
        }
        
        
    }

//binary serach on workable solution
class Solution {
    public static int vertexCover(int n, int m, int[][] edges) {
        // code here
        /*
        binary search on workable solution paradigm
        need to represent graphs as adj matrix size, n by n, indicating edge between some node i and some node j
        represent candidate covering set as int (bit masking)
        notes on getting the next combindation (another way would be to try all combinations of a certain size)
            note we would need to check all combins of some size k before reducing seach space
            i would have precompute all possbile sets of size k, then checked them all, but this is a new way of gereating the next set on the fly
        
        find least signigicant bit as set & - set
        adding c to set to get the next combindation; r = set + c
        reverse the bits from the least signiigcant bit to the end  r ^ set
        finally divide the result by c and bitwise OR to obtain next combination
        */
        //build graph
        int graph[][] = new int[n][n];
        for (int i = 0; i < m; i++){
            graph[edges[i][0] - 1][edges[i][1] - 1] = 1;
            graph[edges[i][1] - 1][edges[i][0] - 1] = 1;   
        }
        
        return findCover(n,m,graph);
    }
    public static int findCover(int n, int m, int[][] g){
        int low = 1;
        int high = n;
        
        while (low < high){
            int mid = low + (high - low) / 2;
            if (checkEdges(n,mid,m,g) == false){
                low = mid + 1;
            }
            else{
                high = mid;
            }
        }
        
        return low;
    }
    public static boolean checkEdges(int n, int k, int m, int[][] g){
        //k is the size of the candidate vertex cover set we wish to check
        int set = (1 << k) - 1;
        int limit = 1 << n;
        
        while (set < limit){
            boolean visited[][] = new boolean[n][n];
            int count = 0;
            for (int i = 1, j = 1; i < limit; i = i << 1, j++){
                if ((set & i) != 0){
                    //add
                    for (int v = 0; v < n; v++){
                        if (g[j-1][v] == 1 && !visited[j-1][v]){
                            visited[j-1][v] = true;
                            visited[v][j-1] = true;
                            count++;
                        }
                    }
                }
            }
            if (count == m){
                return true;
            }
            //find new set
            int co= set & -set;
            int ro= set + co;
            set= (((ro^set)>>2)/co)|ro;
        }
        return false;
    }
}