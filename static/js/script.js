addResponseMsg('Hola')

function addResponseMsg(msg) {
    var div = document.createElement("div");
    div.innerHTML = "<div class='chat-message'>" + msg + "</div>"
    div.className = "chat-message-div"
    document.getElementById("message-box").appendChild(div)
    running = false
}


