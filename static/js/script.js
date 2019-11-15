(function($) {

    const PAGE_SIZE = 10;

    let myMap, resultItem, searchResult, resultsBlock,
        currentPage, resultsNavBtns, resultsActiveNavBtn;

    $(() => {
        const header = $('#header');
        const checkScroll = () => {
            const scrolled = window.pageYOffset || document.documentElement.scrollTop;
            if (header.is('.scrolled') && (scrolled < 50)) {
                header.removeClass('scrolled');
            } else if (!header.is('.scrolled') && (scrolled > 50)) {
                header.addClass('scrolled');
            }
        };
        window.onscroll = () => {
            checkScroll();
        };
        checkScroll();

        resultItem = $($('#result-item-tpl').html());
        resultsBlock = $('#visible-results');
    });


    let mapCenter, placemark;
    const formRequest = {};

    const calculateFormRequest = {};


    const checkMetroDuration = () => {
        window['ymaps'].geocode(mapCenter, { kind: 'metro' }).then((res) => {
            const nearest = res.geoObjects.get(0);
            const multiRouteModel = new window['ymaps'].multiRouter.MultiRouteModel([
                mapCenter,
                nearest.geometry.getCoordinates()
            ], {
                routingMode: 'pedestrian'
            });
            multiRouteModel.events.add("requestsuccess", () => {
                const routes = multiRouteModel.getRoutes();
                calculateFormRequest.time_to_metro = Math.round(routes[0].properties.get("duration").value / 60);
            });
        });
    };


    const setMarkerPosition = () => {
        placemark.geometry.setCoordinates(mapCenter);
        calculateFormRequest.lat = mapCenter[0];
        calculateFormRequest.lng = mapCenter[1];
        checkMetroDuration()
    };


    const init = () => {
        mapCenter = [55.76, 37.64];
        myMap = new window['ymaps'].Map("realty-map", {
            center: mapCenter,
            zoom: 15
        });
    };


    const iniMarkerPosition = (event) => {
        mapCenter = event.get('coords');
        setMarkerPosition();
    };

    const iniSearchForm = () => {
        myMap.events.remove('click', iniMarkerPosition);
        myMap.geoObjects.remove(placemark);
    };

    const iniAnalyzeForm = () => {
        placemark = new window['ymaps'].Placemark(mapCenter, {}, {
            preset: 'islands#redIcon'
        });
        myMap.geoObjects.add(placemark);
        setMarkerPosition();
        myMap.events.add('click', iniMarkerPosition);
    };



    window['ymaps'].ready(init);


    $.fn.validator = function() {
        $(this).each(function() {
            const input = $(this);
            input.on("input keydown keyup mousedown mouseup select contextmenu drop", function(event) {

            });
        });
    };

    const showResultPage = function(pageNumber) {
        resultsBlock.html('');
        currentPage = pageNumber;

        resultsActiveNavBtn ? resultsActiveNavBtn.removeClass('active') : false;
        resultsActiveNavBtn = resultsNavBtns[pageNumber].addClass('active');

        const visibleItems = searchResult.slice(PAGE_SIZE * pageNumber, PAGE_SIZE * pageNumber + PAGE_SIZE);
        $('#visible-items-count').text(visibleItems.length);

        visibleItems.forEach((oneResultItem) => {

            const resultItemElement = resultItem.clone();
            oneResultItem.price_per_m = Math.round((oneResultItem.price / oneResultItem.full_sq));

            if (oneResultItem.metros && oneResultItem.metros.length) {
                oneResultItem.metro = oneResultItem.metros[0].station;
                oneResultItem.time_to_metro = oneResultItem.metros[0].time_to_metro;
            } else {
                oneResultItem.metro = undefined;
                oneResultItem.time_to_metro = undefined;
                oneResultItem.metro_hidden = "true";
            }

            for (let param in oneResultItem) {
                const valElement = resultItemElement.is('[data-' + param + ']') ? resultItemElement : resultItemElement.find('[data-' + param + ']');
                if (valElement.length) {
                    if (param === 'price' || param === 'price_per_m') {
                        oneResultItem[param] = oneResultItem[param].toLocaleString();
                    }
                    switch (valElement.data(param)) {
                        case 'text':
                            valElement.text(oneResultItem[param]);
                            break;
                        default:
                            valElement.attr(valElement.data(param), oneResultItem[param]);
                    }
                }
            }
            resultsBlock.append(resultItemElement);
        })
    };


    $(() => {
        const results = $('#results');
        const resultPrice = $('#result_price');
        const resultDuration = $('#result_duration');


        $('#search-init').on('click', () => {
            iniSearchForm();
        });
        $('#analyze-init').on('click', () => {
            iniAnalyzeForm();
        });
        $('[data-validator]').validator();




        const searchForm = $('#search-form').on('submit', (event) => {
            event.preventDefault();
            const fields = $('input, select', searchForm);
            const values = {};
            fields.each(function() {
                const itemModel = $(this);
                const itemType = itemModel.attr('type');
                if (itemModel.attr('name')) {
                    switch (itemType) {
                        case 'radio':
                            values[itemModel.attr('name')] = itemModel.is(':checked') ? itemModel.val() : values[itemModel.attr('name')] || undefined;
                            break;
                        case 'checkbox':
                            values[itemModel.attr('name')] = itemModel.is(':checked') ? '1' : '0';
                            break;
                        default:
                            values[itemModel.attr('name')] = itemModel.val() || undefined;
                    }
                }
            });
            const convertedFormData = values;
            for (const k in convertedFormData) {
                formRequest[k] = convertedFormData[k];
            }
            results.hide();
            const mapInfo = myMap.getBounds();
            formRequest['latitude_from'] = mapInfo[0][0];
            formRequest['latitude_to'] = mapInfo[1][0];
            formRequest['longitude_from'] = mapInfo[0][1];
            formRequest['longitude_to'] = mapInfo[1][1];

            const resultsBlock = $('#results-list-block');
            resultsBlock.hide();

            const navigate = $('#results-navigate');
            navigate.html('');

            $.ajax({
                url: '/api/mean/',
                // url: 'static/test.json',
                data: formRequest
            }).always(() => {
                // results.show();
            }).then((res) => {
                searchResult = [];
                for (let k in res.flats) {
                    searchResult.push(res.flats[k]);
                }
                if (searchResult.length) {
                    resultsBlock.show();
                }
                resultsNavBtns = [];
                $('#search-items-count').text(searchResult.length);
                const pagesCount = Math.ceil(searchResult.length / PAGE_SIZE);
                for (let n = 0; n < pagesCount; n++) {
                    const navButton = $('<a>');
                    navButton.attr('href', '#results-list');
                    navButton.addClass('results-list_navigation-btn');
                    navButton.text(n + 1);
                    navButton.data('page', n);
                    navButton.on('click', () => {
                        showResultPage(navButton.data('page'));
                    });
                    resultsNavBtns.push(navButton);
                    navigate.append(navButton);
                }
                showResultPage(0);
            });
            return false;
        });

        const calculateForm = $('#calculate-form').on('submit', (event) => {
            event.preventDefault();
            const fields = $('input, select', calculateForm);
            const values = {};

            fields.each(function() {
                const itemModel = $(this);
                const itemType = itemModel.attr('type');
                if (itemModel.attr('name')) {
                    switch (itemType) {
                        case 'checkbox':
                            values[itemModel.attr('name')] = itemModel.is(':checked') ? '1' : '0';
                            break;
                        default:
                            values[itemModel.attr('name')] = itemModel.val() || undefined;
                    }
                }
            });

            const convertedFormData = values;

            for (const k in convertedFormData) {
                calculateFormRequest[k] = convertedFormData[k];
            }

            results.hide();

            $.ajax({
                url: '/map',
                data: calculateFormRequest
            }).always(() => {
                results.show();
            }).then((res) => {
                resultPrice.text(res.Price + ' руб.');
                resultDuration.text(res.Duration + ' дн.');
            });
            return false;
        });

    });




})(jQuery);
