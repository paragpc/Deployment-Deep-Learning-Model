$(document).ready(function () {
    $('.select-image-result').hide();
    $('.wheel').hide();
    $('#result').hide();

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#preview-image').css('background-image', 'url(' + e.target.result + ')');
                $('#preview-image').hide();
                $('#preview-image').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        $('.select-image-result').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        $(this).hide();
        $('.wheel').show();
        $.ajax({
            type: 'POST',
            url: '/predict_yoga_pose',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('.wheel').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Yoga Pose:  ' + data);
                console.log('Predicted the yoga pose');
            },
        });
    });

});
