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