$(document).ready(function(){
    $('#typed_text').focus();
})

function displayReceivedMsg(text, time)
{
    var code = 
    '<div class="incoming_msg">\
        <div class="incoming_msg_img">\
            <img src="https://ptetutorials.com/images/user-profile.png"/>\
        </div>\
        <div class="received_msg">\
            <div class="received_withd_msg">\
                <p>'+text+'</p>\
                <span class="time_date"> '+time+'</span>\
            </div>\
        </div>\
    </div>'

    $('#msg_hist').append(code)
}


function displaySentMsg(text, time)
{
    var code = 
    '<div class="outgoing_msg">\
        <div class="sent_msg">\
            <p>'+text+'</p>\
            <span class="time_date">'+time+'</span>\
        </div>\
    </div>'

    $('#msg_hist').append(code)
}

function sendMsg()
{
    URL_PATH = "http://192.168.0.102:8000/bot/"
    msg = $('#typed_text').val();
    $('#typed_text').val('');
    displaySentMsg(msg, new Date().toLocaleString());
    $.ajax({
        type: 'POST',
        url: URL_PATH,
        contentType: 'application/json',
        data: JSON.stringify({ "text": msg }),
        success: function(data) {
            displayReceivedMsg(data.response, new Date().toLocaleString());
        }
    })
}

$(document).on('keypress',function(e) {
    if(e.which == 13) {
        sendMsg();
    }
});
