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
