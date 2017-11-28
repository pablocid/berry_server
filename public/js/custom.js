/***************************************************************************************************************
||||||||||||||||||||||||||||         CUSTOM SCRIPT FOR Digital Media        ||||||||||||||||||||||||||||||||||||
****************************************************************************************************************
||||||||||||||||||||||||||||              TABLE OF CONTENT                  ||||||||||||||||||||||||||||||||||||
****************************************************************************************************************
****************************************************************************************************************
1. menuzordActive 
2. bootstrapAnimatedLayer
3. owlActive 
4. shareCollapse 
5. contactFormValidation 
6. galleryMasonaryLayout
7. stickyHeader 
****************************************************************************************************************
||||||||||||||||||||||||||||            End TABLE OF CONTENT                ||||||||||||||||||||||||||||||||||||
****************************************************************************************************************/


"use strict"; // Start of use strict

function menuzordActive () {
    if ($("#menuzord").length) {
        $("#menuzord").menuzord({
            indicatorFirstLevel: '<i class="fa fa-angle-down"></i>'
        });
    };
}

function bootstrapAnimatedLayer() {

    /* Demo Scripts for Bootstrap Carousel and Animate.css article
     * on SitePoint by Maria Antonietta Perna
     */

    //Function to animate slider captions 
    function doAnimations(elems) {
        //Cache the animationend event in a variable
        var animEndEv = 'webkitAnimationEnd animationend';

        elems.each(function() {
            var $this = $(this),
                $animationType = $this.data('animation');
            $this.addClass($animationType).one(animEndEv, function() {
                $this.removeClass($animationType);
            });
        });
    }

    //Variables on page load 
    var $myCarousel = $('#minimal-bootstrap-carousel'),
        $firstAnimatingElems = $myCarousel.find('.item:first').find("[data-animation ^= 'animated']");

    //Initialize carousel 
    $myCarousel.carousel({
        interval: 7000
    });

    //Animate captions in first slide on page load 
    doAnimations($firstAnimatingElems);

    //Pause carousel  
    $myCarousel.carousel('pause');


    //Other slides to be animated on carousel slide event 
    $myCarousel.on('slide.bs.carousel', function(e) {
        var $animatingElems = $(e.relatedTarget).find("[data-animation ^= 'animated']");
        doAnimations($animatingElems);
    });
}

function owlActive () {
    if ($('.client-carousel .owl-carousel').length) {
        $('.client-carousel .owl-carousel').owlCarousel({
            loop: true,
            nav: false,
            dots: false,
            autoWidth: false,
            autoplay:true,
            autoplayTimeout:5000,
            autoplayHoverPause:true,
            margin: 90,
            responsiveClass: true,
            responsive: {
                0: {
                    items: 1,
                    autoWidth: true,
                    margin: 0
                },
                320: {
                    items: 1,
                    autoWidth: true,
                    margin: 40
                },
                480: {
                    items: 3,
                    autoWidth: true,
                    margin: 40
                },
                600: {
                    items: 4
                },
                1000: {
                    items: 6
                }
            }
        });
    };
    if ($('.team-carousel').length) {
        $('.team-carousel').owlCarousel({
            loop: true,
            nav: false,
            navText: [
                '<i class="fa fa-angle-left"></i>',
                '<i class="fa fa-angle-right"></i>'
            ],
            dots: true,
            autoplay:true,
            autoplayTimeout:5000,
            autoplayHoverPause:true,
            margin: 30,
            responsiveClass: true,
            responsive: {
                0: {
                    items: 1,
                    dots: false,
                    nav: true
                },
                480: {
                    items: 2,
                    dots: false,
                    nav: true
                },
                600: {
                    items: 3
                },
                1000: {
                    items: 4
                }
            }
        });
    };
}

function shareCollapse () {
    if ($('.share-collapse').length) {
        $('.share-collapse').on('click', function () {
            $(this).parent('.share-box').find('.list-inline').toggleClass('slideOutLeft slideInLeft');
            return false;
        });
    };
}
function contactFormValidation () {

    if($('.validate-cf').length){
        $('.validate-cf').validate({ // initialize the plugin
            rules: {
                name: {
                    required: true
                },
                email: {
                    required: true,
                    email: true
                },
                message: {
                    required: true
                },
                subject: {
                    required: true
                }
            },
            submitHandler: function (form) { 
                // sending value with ajax request
                $.post($(form).attr('action'), $(form).serialize(), function (response) {
                    $(form).parent('div').append(response);
                    $(form).find('input[type="text"]').val('');
                    $(form).find('input[type="email"]').val('');
                    $(form).find('textarea').val('');
                });
                return false;
            }
        });
    }
}

function galleryMasonaryLayout() {
    if ($('.masonary-layout').length) {
        $('.masonary-layout').isotope({
            layoutMode: 'masonry'
        });
    }

    if ($('.post-filter').length) {
        $('.post-filter li').children('span').on('click', function() {
            var Self = $(this);
            var selector = Self.parent().attr('data-filter');
            $('.post-filter li').children('span').parent().removeClass('active');
            Self.parent().addClass('active');
            $('.filter-layout').isotope({
                filter: selector,
                stagger: 30,
                transitionDuration: 400,
                hiddenStyle: {
                    opacity: 0
                },
                visibleStyle: {
                    opacity: 1
                }
            });
            return false;
        });
    }

    if ($('.post-filter.has-dynamic-filter-counter').length) {
        // var allItem = $('.single-filter-item').length;

        var activeFilterItem = $('.post-filter.has-dynamic-filter-counter').find('li');

        activeFilterItem.each(function() {
            var filterElement = $(this).data('filter');
            console.log(filterElement);
            var count = $('.gallery-content').find(filterElement).length;

            $(this).children('span').append('<span class="count"><b>' + count + '</b></span>');
        });
    };
}

function stickyHeader () {
    if ($('.stricky').length) {
        var strickyScrollPos = 100;
        if($(window).scrollTop() > strickyScrollPos) {
            $('.stricky').removeClass('fadeIn animated');
            $('.stricky').addClass('stricky-fixed fadeInDown animated');
            $('.scroll-to-top').fadeIn(500);
        }
        else if($(this).scrollTop() <= strickyScrollPos) {
            $('.stricky').removeClass('stricky-fixed fadeInDown animated');
            $('.stricky').addClass('slideIn animated');
            $('.scroll-to-top').fadeOut(500);
        }
    };
}

// Dom Ready Function
jQuery(document).on('ready', function () {
    (function ($) {
        // add your functions
        menuzordActive();
        bootstrapAnimatedLayer();
        owlActive();
        shareCollapse();
        contactFormValidation();
    })(jQuery);
});
// window on load functino
jQuery(window).on('load', function () {
    (function ($) {
        // add your functions
        galleryMasonaryLayout();
    })(jQuery);
});
// window on scroll functino
jQuery(window).on('scroll', function () {
    (function ($) {
        // add your functions
        stickyHeader();
    })(jQuery);
});