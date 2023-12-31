/*
 * Binary Tree to CDL
 * 16NOV23
 */
//User function Template for Java
/*
Node defined as
class Node{
    int data;
    Node left,right;
    Node(int d){
        data=d;
        left=right=null;
    }
}
*/
class Solution
{ 
    //just use global, in python can call self
    //keep prev and head pointers, we will head as the answer
    Node head = null;
    Node prev = null;
    //Function to convert binary tree into circular doubly linked list.
    Node bTreeToClist(Node root)
    {
        //your code here
        inorder(root);
        return head;
    }
    public void inorder(Node node){
        if (node == null){
            return;
        }
        
        inorder(node.left);
        //first assignment
        if (prev == null){
            head = node;
        }
        //oother move
        else{
            //prev's next should point to the current node
            prev.right = node;
            //currenot node should point to prev
            node.left = prev;
        }
        //aftet finishing move prev
        prev = node;
        inorder(node.right);
        //close the loop
        head.left = prev;
        prev.right = head;
    }
    
}

//keep flag variable
//in java need to use arrays to simulat pass by reference
class Solution {
    Node last;

    // Function to convert binary tree into circular doubly linked list.
    void solve(Node root, Node[] head, Node[] prev, int[] flag) {
        if (root == null)
            return;

        solve(root.left, head, prev, flag);

        if (flag[0] == 0) {
            flag[0] = 1;
            head[0] = root;
            prev[0] = root;
        } else {
            prev[0].right = root;
            prev[0].right.left = prev[0];
            last = root;
            prev[0] = prev[0].right;
        }

        solve(root.right, head, prev, flag);
    }

    Node bTreeToCList(Node root) {
        // Base case
        Node[] head = new Node[1];
        Node[] prev = new Node[1];
        int[] flag = {0};
        solve(root, head, prev, flag);
        head[0].left = last;
        last.right = head[0];
        return head[0];
    }
}

/*
 *  Reverse a Doubly Linked List

 */
class Solution{


    public static Node reverseDLL(Node  head)
    {
        //Your code here
        Node curr = head;
        Node temp = head.next;
        
        while (temp != null){
            swap(curr);
            curr = temp;
            temp = temp.next;
        }
        
        swap(curr);
        return curr;
    }


    public static void swap(Node node){
        //keep reference to prev node before swapping
        Node temp = node.prev;
        node.prev = node.next;
        node.next = temp;
    }
}

class Solution{


    //Function to reverse a doubly linked list
    public static Node reverseDLL(Node head)
    {
        //checking if head is null or head.next is null,
        //if true, then return the head itself as there is no need to reverse a list with 0 or 1 node
        if(head == null || head.next == null)
            return head;
        
        //declaring a current and previous node
        Node curr = head, prev = null;
        
        //looping through the list
        while(curr != null){
            //storing the previous node in prev
            prev = curr.prev;
            //swapping the prev and next pointers of the current node
            curr.prev = curr.next;
            curr.next = prev;
            
            //moving the current node to its previous node
            curr = curr.prev;
        }
        //returning the previous node of head as the new head (the last node after reversing the list)
        return prev.prev;

}

/*
 * Intersection of two sorted Linked lists
 */
class Solution
{
   public static Node findIntersection(Node head1, Node head2)
    {
        // code here.
        /*
        dummy list and add to list if they are equal, otherwise move the smaller one
        */
        Node dummy = new Node(-1);
        Node curr = dummy;
        
        while ((head1 != null) && (head2 != null)){
            //match
            if (head1.data == head2.data){
                Node temp = new Node(head1.data);
                curr.next = temp;
                curr = curr.next;
                head1 = head1.next;
                head2 = head2.next;
            }
            
            else if (head1.data < head2.data){
                head1 = head1.next;
            }
            
            else{
                head2 = head2.next;
            }
        }
        
        return dummy.next;
    }
}

/*
 * K Sum Paths
keep path in list, and every time go backwards from the path, and when sum == k, this is a path from the current ndoe to tned of the path

 */

class Solution
{
    int count = 0;
    public int sumK(Node root,int k)
    {
        // code here 
        ArrayList<Integer> path = new ArrayList<>();
        dp(root,k,path);
        return count;
    }
    
    void dp(Node node, int k, ArrayList<Integer> path){
        if (node == null){
            return;
        }
        
        path.add(node.data);
        int sum = 0;
        //go backwards
        for (int i = path.size() - 1; i >= 0; i--){
            sum += path.get(i);
            if (sum == k){
                count++;
            }
        }
        
        dp(node.left,k,path);
        dp(node.right,k,path);
        //backtrack
        path.remove(path.size() - 1);
    }
}

class Solution
{
    int mod = 1_000_000_007;
    int ans = 0;
    HashMap<Integer,Integer> mapp = new HashMap<>();
    public int sumK(Node root,int k)
    {
        /*
        we can use prefix sum, but with tree insteaf
        maintain running sum as we go down the tree and at each step, check whether the difference between running sum and k has been seen before
        we can use hashmap to store counts of path sums
        */
        dp(root,k,0);
        return ans;
    }
    
    public void dp(Node root, int needed_sum, int curr_sum){
        if (root == null){
            return;
        }
        
        //find complement
        //mapp is global
        int complement = curr_sum + root.data - needed_sum;
        ans = (ans + mapp.getOrDefault(complement,0)) % mod;
        ans %= mod;
        
        //not only check for complement, but also the current path
        if ((curr_sum + root.data) == needed_sum){
            ans += 1;
            ans %= mod;
        }
        
        //update mapp with curr sum
        mapp.put(curr_sum + root.data, (mapp.getOrDefault(curr_sum + root.data,0) + 1) % mod);
        
        //recurse
        dp(root.left,needed_sum, curr_sum + root.data);
        dp(root.right,needed_sum, curr_sum + root.data);
        
        //backtrack
        mapp.put(curr_sum + root.data, (mapp.getOrDefault(curr_sum + root.data,0) - 1) % mod);
    }
}

//dont forget prefix sum and compelment sum with binary trees


/*
 * Determine if Two Trees are Identical
 */
class Solution
{
    //Function to check if two trees are identical.
	boolean isIdentical(Node root1, Node root2)
	{
	    /*
	    dp(p,q) returnes where nodes p and q are identical
	    
	    */
	    return dp(root1,root2);
	}
	
	boolean dp(Node p, Node q){
	    //if one is empty and the other is not return false;
	    if (p == null || q == null){
	        return p == q;
	    }
	    
	    if (p == null && q == null){
	        return true;
	    }
	    
	    if (p.data != q.data){
	        return false;
	    }
	    
	    return dp(p.left,q.left) && dp(p.right,q.right);
	    
	}
	
}

class Solution
{
    //Function to check if two trees are identical.
	boolean isIdentical(Node root1, Node root2)
	{
	    // Code Here
	    /*
	    if bothe trees are empty, they are the same
	    if non empty, check nodes are same and simlar lareft and right
	    */
	    
	    return dp(root1,root2);
	}
	
	boolean dp(Node p, Node q){
	    if (p == null && q == null){
	        return true;
	    }
	    
	    else if (p == null && q != null){
	        return false;
	    }
	    //essentially have (p and not q) or (not p and q), which is just (p ^ q), mutually exclusive OR
	    else if (p != null && q == null){
	        return false;
	    }
	    
	    return (p.data == q.data) && dp(p.left,q.left) && dp(p.right,q.right);
	}
	
}

/*
 * 
Symmetric Tree
 */

 // complete this function
// return true/false if the is Symmetric or not
class GfG
{
    // return true/false denoting whether the tree is Symmetric or not
    public static  boolean isSymmetric(Node root)
    {
        // add your code here;
        /*
        let dp(node) return whether or not a tree rooted is symetric
        for the call at the root we we need dp(node.left) && dp(nodde.right)
        empty node is trivally symmtric
        need to pass in dp(node,node) for the same root
        */
        
        return dp(root,root);
    }
    
    public static boolean dp(Node p, Node q){
        //empty cases, there are three
        if (p == null && q == null){
            return true;
        }
        
        else if (p != null && q == null){
            return false;
        }
        
        else if (p == null & q != null){
            return false;
        }
        
        else if (p.data != q.data){
            return false;
        }
        //need to check (left,right) and (right,left)
        boolean left_right = dp(p.left,q.right);
        boolean right_left = dp(p.right,q.left);
        return left_right && right_left;
    }
}

/*
 * AVL Tree Insertion
 */
class Solution
{
    public  Node insertToAVL(Node node,int data)
    {
        //code here
        if (node == null) {
            return new Node(data);
        }
        if (node.data > data) {
            node.left = insertToAVL(node.left, data);
        } else if (node.data < data) {
            node.right = insertToAVL(node.right, data);
        }
        node.height = Math.max(height(node.left), height(node.right)) + 1;
        int diff = height(node.left) - height(node.right);
        if (diff > 1) { // Left subtree is higher
            if (data < node.left.data) {
                return rightRotate(node);
            } else if (data > node.left.data) { // LR rotation
                node.left = leftRotate(node.left);
                return rightRotate(node);
            }
        } else if (diff < -1) { // Right subtree is higher
            if (data < node.right.data) { // RL rotation
                node.right = rightRotate(node.right);
                return leftRotate(node);
            } else if (data > node.right.data) {
                return leftRotate(node);
            }
        }
        return node;
    }
    
    
    
    public Node leftRotate(Node a) {
        Node b = a.right;
        Node t2 = b.left;
        b.left = a;
        a.right = t2;
        a.height = Math.max(height(a.left), height(a.right)) + 1;
        b.height = Math.max(height(b.left), height(b.right)) + 1;
        return b;
    }
    
    public Node rightRotate(Node a) {
        Node b = a.left;
        Node t2 = b.right;
        b.right = a;
        a.left = t2;
        a.height = Math.max(height(a.left), height(a.right)) + 1;
        b.height = Math.max(height(b.left), height(b.right)) + 1;
        return b;
    }

    //get height util
    public int height (Node node){
        if (node == null){
            return 0;
        }
        
        return node.height;
    }
} 

/*
 * Shuffle Array
 */
class  Solution
{
    void shuffleArray(long arr[], int n)
    {
        // Your code goes here
        /*
        split into two arrays and combein
        a is left pair b is right pair
        */
        long[] a = new long[n/2];
        long[] b = new long[n/2];
        
        for (int i = 0; i < n / 2; i++){
            a[i] = arr[i];
            b[i] = arr[i + n/ 2];
        }
        
        int j = 0;
        for (int i = 0; i < n; i += 2){
            arr[i] = a[j];
            arr[i+1] = b[j];
            j++;
            
        }
        

        //System.out.println(Arrays.toString(a));
        //System.out.println(Arrays.toString(b));
        
        
    }
}

/*
 * Print Pattern
 */

 class Solution{
    public List<Integer> pattern(int N){
        // code here
        /*
        need to use recursino
        forward and revers direction
        */
        ArrayList<Integer> ans = new ArrayList<>();
        int first_end = decrement(N,ans);
        int second_end = increment(first_end, N,ans);
        ans.add(second_end);
        return ans;
    }
    public int decrement(int N, ArrayList<Integer> ans){
        
        if (N <= 0){
            return N;
        }
        
        ans.add(N);
        return decrement(N-5,ans);
    }
    
    public int increment(int N, int limit, ArrayList<Integer> ans){
        if (N >= limit){
            return N;
        }
        
        ans.add(N);
        return increment(N+5,limit,ans);
    }
}

//all in one function
class Solution{
    public List<Integer> pattern(int N){
        // code here
        /*
        need to use recursino
        forward and revers direction
        */
        ArrayList<Integer> ans = new ArrayList<>();
        decrement(N,ans);
        return ans;
    }
    public void decrement(int N, ArrayList<Integer> ans){
        
        if (N <= 0){
            ans.add(N);
            return;
        }
        
        ans.add(N);
        decrement(N-5,ans);
        ans.add(N);
    }
    

}

/*
 * Detect Cycle using DSU
 */

class DSU
{
    int[] parent;
    int[] size;
    
    public DSU(int n){
        parent = new int[n];
        size = new int[n];
        Arrays.fill(size,1);
        //fill in sizes and self points
        for (int i = 0; i < n; i++){
            parent[i] = i;
        }
    }
    
    public int find(int x){
        if (parent[x] == x){
            return parent[x];
        }
        
        parent[x] = find(parent[x]);
        return parent[x];
        
    }
    
    public void union(int x, int y){
        //find parents
        int x_par = find(x);
        int y_par = find(y);
        
        if (x_par == y_par){
            return;
        }
        //bigger guy
        if (size[x_par] > size[y_par]){
            parent[y_par] = x_par;
            size[x_par] += size[y_par];
        }
        //smalle guy, dont worry about shifting ranks
        else{
            parent[x_par] = y_par;
            size[y_par] += size[x_par];
        }
        
        
    }
}
class Solution
{
    //Function to detect cycle using DSU in an undirected graph.
    public int detectCycle(int V, ArrayList<ArrayList<Integer>> adj)
    {
        // Code here
        /*
        adj readhs node : {neigh nodes} this is a list
        i dont need to rebuild
        need to use DSU, if two nodes are already belong to the same parent, there is a cycle
        when joining an edge, if the two nodes already belong to the same parent, then there is a cycle
        just check find
        */
        DSU dsu = new DSU(V);
        Set<String> set = new HashSet<>();
        for (int node = 0; node < V; node++){
            for (int neigh : adj.get(node)){
                if (set.contains(helper(node,neigh))){
                    continue;
                }
                
                if (dsu.find(node) != dsu.find(neigh)){
                    dsu.union(node,neigh);
                    set.add(helper(node,neigh));
                    set.add(helper(neigh,node));
                }
                else{
                    return 1;
                }
            }
        }
        return 0;
    }
    
    //use to hash an edge
    public String helper(int a, int b){
        String ans = Integer.toString(a);
        ans += Integer.toString(b);
        return ans;
    }
}

/*
 * Sum of Dependencies
 */

 class Solution {
    int sumOfDependencies(ArrayList<ArrayList<Integer>> adj, int V) {
        // code here
        /*
        depdencies are just in degree for a node
        keep track of the indegree incounts array,
        input it not an edge list, but an adjaceny list
        */
        int[] indegree = new int[V];
        
        for (ArrayList<Integer> neighs : adj){
            for (int neigh : neighs){
                indegree[neigh]++;
            }
        }
        
        int ans = 0;
        for (int ind : indegree){
            ans += ind;
        }
        return ans;
    }
};

class Solution {
    int sumOfDependencies(ArrayList<ArrayList<Integer>> adj, int V) {
        // code here
        /*
        we can also do just the size
        DUH!
        */
        int ans = 0;
        
        for (ArrayList<Integer> neighs : adj){
            ans += neighs.size();
        }
        

        return ans;
    }
};


/*
 * Euler circuit and Path
 */
class Solution{
    public int isEulerCircuit(int V, List<Integer>[] adj) 
    {
        /*
        eulerian path, visits every edge once, and ends up on different nodes
        eulerian circuit, is an eulerian path but stats and ends on the sam vertex
        
        i.e a grpah will have an euler circuit if and only iff all degrees are even
        a graph will contain an eiler path if and olf if it contains exactly two odd verticies
        */
        ArrayList<Integer> sizes = new ArrayList<>();
        
        int evendegrees = 0;
        int odddegrees = 0;
        
        for (int i = 0; i < V; i++){
            sizes.add(adj[i].size());
        }
        
        for (int i = 0; i < V; i++){
            if (sizes.get(i) % 2 == 0){
                evendegrees++;
            }
            else{
               odddegrees++; 
            }
        }
        
        if (evendegrees == V){
            return 2;
        }
        else if (odddegrees > 0 && odddegrees == 2 ){
            return 1;
        }
        return 0;
    }
}

/*
 * Check whether BST contains Dead End
 */

 class Solution
{
    public static boolean isDeadEnd(Node root)
    {
        //Add your code here.
        /*
        formal description, a leaf node is a node having value val and if nodes val+1 and val-1 already exsist in the tree
        however for a node with value 1, we call it a Dead End, so we check for 2, one way to do it is hash all the nodes and check leaf end condidtions
        */
        HashSet<Integer> seen = new HashSet<>();
        dfs(root,seen);
        return check(root,seen);
    }
    
    public static void dfs(Node node, HashSet<Integer> seen){
        if (node == null){
            return;
        }
        
        seen.add(node.data);
        dfs(node.left,seen);
        dfs(node.right, seen);
    }
    
    //check
    public static boolean check(Node node, HashSet<Integer> seen){
        //check conditions
        if (node.left == null && node.right == null){
            if (node.data == 1){
                if (seen.contains(2)){
                    return true;
                }
            }
            
            else if (node.data == 10001){
                if (seen.contains(1000)){
                    return true;
                }
            }
            
            else{
                if (seen.contains(node.data - 1) && seen.contains(node.data + 1)){
                    return true;
                }
            }
            
        }
        
        boolean left = (node.left != null) ? check(node.left,seen) : false;
        boolean right = (node.right != null) ? check(node.right,seen) : false;
        return left || right;

    }
}

//Function should return true if a deadEnd is found in the bst otherwise return false.
class Solution
{
    public static boolean isDeadEnd(Node root)
    {
        //Add your code here.
        //another way is to just check if for a leaf node, we have x - 1 and x + 1 are in the trees
        if (root == null){
            return false;
        }
        return find(root, 0,Integer.MAX_VALUE);
    }
    public static boolean find(Node node, int MIN, int MAX){
        if (node == null){
            return false;
        }
        
        if ((node.data - MIN == 1) && (MAX - node.data == 1)){
            return true;
        }
        
        boolean left = find(node.left, MIN, node.data);
        boolean right = find(node.right, node.data, MAX);
        if (left || right){
            return true;
        }
        return false;
    }
}