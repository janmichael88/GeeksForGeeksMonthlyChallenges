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
