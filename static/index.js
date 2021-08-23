(function() {
    var selectmodel = document.querySelector("#selectModel");
	var canvas = document.querySelector("#canvas");
	var context = canvas.getContext("2d");
	canvas.width = 280;
	canvas.height = 280;
	var mousePos = { x: 0, y: 0 };
	var lastMousePos = { x: 0, y: 0 };
	context.fillStyle = "black";
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.color = "white";
	context.lineWidth = 15;
	context.lineJoin = context.lineCap = "round";
	var isDrawing = false;
	
	clearCanvas();

	canvas.addEventListener("mousedown",
							function(e) {
								canvas.addEventListener("mousemove", onDraw, false);
								isDrawing = 1;
							}, 
							false);
							
	canvas.addEventListener("touchstart",
							function(e) {
								e.preventDefault();
								mousePos = getTouchPos(canvas, e);
								lastMousePos.x = mousePos.x;
								lastMousePos.y = mousePos.y;								
								canvas.addEventListener("touchmove", onDraw, false);
								isDrawing = 1;
							}, 
							false);							
							
							
	canvas.addEventListener("mouseup",
							function() {
								canvas.removeEventListener("mousemove", onDraw, false);
								isDrawing = 0;
							}, 
							false);
							
	canvas.addEventListener("touchend",
							function() {
								canvas.removeEventListener("touchmove", onDraw, false);
								isDrawing = 0;
							}, 
							false);									
							
	canvas.addEventListener("mousemove",
							function(e) { 
								lastMousePos.x = mousePos.x;
								lastMousePos.y = mousePos.y;
								// mousePos is relative to the top-left corner
								mousePos.x = e.pageX - this.offsetLeft;
								mousePos.y = e.pageY - this.offsetTop;
							}, 
							false);									
							
	canvas.addEventListener("touchmove",
							function(e) {
								e.preventDefault();
								lastMousePos.x = mousePos.x;
								lastMousePos.y = mousePos.y;								
								mousePos = getTouchPos(canvas, e);
								// $('#result').text(' touchmove: x='+mousePos.x+', y='+mousePos.y);
							}, 
							false);
					
							
	// Get the position of a mobile touch relative to the canvas top-left corner
	function getTouchPos(canvasDom, touchEvent) {
		var rect = canvasDom.getBoundingClientRect();
		return {
			x: touchEvent.touches[0].clientX - rect.left,
			y: touchEvent.touches[0].clientY - rect.top
		};
	}							
	
	/* draw on canvas */
	var onDraw = function() {
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;
		context.beginPath();
		context.moveTo(lastMousePos.x, lastMousePos.y);
		context.lineTo(mousePos.x, mousePos.y);
		context.closePath();
		context.stroke();
	};
	
	
	function clearCanvas() {
		var clearButton = $("#clearButton");
		clearButton.on("click", function(e) {
			e.preventDefault();
			context.clearRect(0, 0, 280, 280);
			context.fillStyle = "black";
			context.fillRect(0, 0, canvas.width, canvas.height);
		});
	
		/* Slider control */
		var slider = document.getElementById("myRange");
		var output = document.getElementById("sliderValue");
		output.innerHTML = slider.value;
		slider.oninput = function() {
			output.innerHTML = this.value;
			context.lineWidth = $(this).val();
		};
		$("#lineWidth").change(function() {
			context.lineWidth = $(this).val();
		});
	}
})();