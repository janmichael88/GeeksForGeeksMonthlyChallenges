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

////////////////////////////////////////////
// Paths from root with a specified sum
// 21JAN24
////////////////////////////////////////////
class Solution
{
    public static ArrayList<ArrayList<Integer>> ans;
    public static ArrayList<ArrayList<Integer>> printPaths(Node root, int sum)
    {
        // code here
        /*
        global dfs, pass path and curr sum along path
        only on leaves we check
        */
        ans = new ArrayList<>();
        ArrayList<Integer> curr_path = new ArrayList<>();
        dfs(root,curr_path,0,sum);
        return ans;
    }
    public static void dfs(Node node, ArrayList<Integer> path, int path_sum, int sum){
        //is leaf
        if (node == null){
            return;
        }
        
        path.add(node.data);
        path_sum += node.data;
        
        if (path_sum == sum){
            ans.add(new ArrayList<>(path));
        }
        if (node.left != null){
            dfs(node.left,path,path_sum,sum);
        }
        if (node.right != null){
            dfs(node.right,path,path_sum,sum);
        }
        path.remove(path.size() - 1); //dont forget to backtrack
    }
}

/////////////////////////////////////////////////////
// Course Schedule
//  22JAN24
/////////////////////////////////////////////////////
class Solution
{
    static int[] findOrder(int n, int m, ArrayList<ArrayList<Integer>> prerequisites) 
    {
        // add your code here
        /*
        this is just kahns algroithm, start with nodes with 0 indegree, then for neighbors dropp edge
        if indegree is zero add node to q
        */
        HashMap<Integer, ArrayList<Integer>> graph = new HashMap<>();
        int[] indegree = new int[n];
        ArrayList<Integer> path = new ArrayList<>();
        for (ArrayList<Integer> edge : prerequisites){
            //(index 1 into index 0) or v into u
            int u = edge.get(0);
            int v = edge.get(1);
            indegree[u]++;
            ArrayList<Integer> neighs = graph.getOrDefault(v, new ArrayList<>());
            neighs.add(u);
            graph.put(v,neighs);
            
        }
        //q up nodes with 0 indegree
        Deque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < n; i++){
            if (indegree[i] == 0)
                q.add(i);
        }

        while (!q.isEmpty()){
            int curr = q.pollFirst();
            //add to ans
            path.add(curr);
            //find neighbors
            for (int neigh : graph.getOrDefault(curr, new ArrayList<>())){
                //remove ege going into neigh
                indegree[neigh]--;
                //if zero add to q
                if (indegree[neigh] == 0)
                    q.add(neigh);
            }
        }
        
        //need to check if ordering is even possible
        for (int i = 0; i < n; i++){
            if (indegree[i] > 0){
                return new int[0];
            }
        }
        int ans[] = path.stream().mapToInt(Integer::intValue).toArray();
        return ans;
    }
}

///////////////////////////////////////////////
// Is it a tree?
// 23JAN24
//////////////////////////////////////////////
class Solution {
    public boolean isTree(int n, int m, ArrayList<ArrayList<Integer>> edges) 
    {
        // code here
        /*
        its a tree is we can touch all the nodes and there isn't a cycle
        there is a coloring algorithm for cycle detection in undirected graph, but do i need to do it here?
        need to pass parent and node
        
        
        */
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++){
            graph.add(new ArrayList<>());
        }
        
        for (ArrayList<Integer> edge : edges){
            int u = edge.get(0);
            int v = edge.get(1);
            graph.get(u).add(v);
            graph.get(v).add(u);
        }
        
        boolean[] seen = new boolean[n];
        dfs(graph,0,seen);
        
        //check we cant touch all
        for (boolean val : seen){
            if (!val){
                return false;
            }
        }
        
        //check for cycle
        Arrays.fill(seen, false);
        return hasCycle(graph,0,-1,seen);
    }
    public void dfs(ArrayList<ArrayList<Integer>> graph, int node, boolean[] seen){
        //mark
        seen[node] = true;
        for (int neigh : graph.get(node)){
            if (!seen[neigh]){
                dfs(graph,neigh,seen);
            }
        }
    }
    
    public boolean hasCycle(ArrayList<ArrayList<Integer>> graph, int node, int parent, boolean[] seen){
        //mark
        seen[node] = true;
        for (int neigh : graph.get(node)){
            if (!seen[neigh]){
                if (!hasCycle(graph,neigh,node,seen)){
                    return false;
                }
            }
            //no back edge
            else if (neigh != parent){
                return false;
            }
        }
        return true;
        
    }
}