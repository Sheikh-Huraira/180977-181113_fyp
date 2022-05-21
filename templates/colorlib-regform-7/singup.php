<?php
$name = $_POST['name'];
$email = $_POST['email'];
$password = $_POST['password'];

if(!empty($name) || !empty($email) || !empty($password)){
	$host = "localhost";
	$dbUsername = "root";
	$dbPassword = "";
	$dbname = "users"


	$conn = new mysqli_connect($host, $dbUsername, $dbPassword, $dbname );
	if (myqli_connect_error()) {
		die('Connect Error('.mysqli_connect_errno().')'.mysqli_connect_error());
		
	}else {

		$SELECT = "SELECT email From register Where email=? LIMIT 1";
		$INSERT = "INSERT Into register(username,email,password) values(?, ?, ?)";

		$stmt = $conn->prepare($SELECT);
		$stmt->bind_param("s", $email);
		$stmt->execute();
		$stmt->bind_result($email);
		$stmt->store_result();
		$rnum= $stmt->num_rows;
		if ($rnum==0) {
			$stmt->close();

			$stmt = $conn->prepare($INSERT);
			$stmt->bind_param("ss",$name,$password);
			$stmt->execute();
			echo "New Data Inserted";
		}else{
			echo "Email already registered";

		}
		$stmt->close();
		$conn->close();


	}




}
else{
	echo "All field are required";
	die();
}

?>