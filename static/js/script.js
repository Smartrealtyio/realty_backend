(function($) {

    const PAGE_SIZE = 10;

    let myMap, resultItem, searchResult, resultsBlock,
        currentPage, resultsNavBtns, resultsActiveNavBtn, clusterer;

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
    let formRequest = {};

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
        clusterer ? myMap.geoObjects.add(clusterer) : clusterer;
    };


    const iniAnalyzeForm = () => {
        placemark = new window['ymaps'].Placemark(mapCenter, {}, {
            preset: 'islands#redIcon'
        });
        myMap.geoObjects.add(placemark);
        setMarkerPosition();
        myMap.events.add('click', iniMarkerPosition);
        clusterer ? myMap.geoObjects.remove(clusterer) : false;
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


        $('#visible-items-count').text(searchResult.flats.length);

        searchResult.flats.forEach((oneResultItem) => {

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
                let visParam;
                if (param === 'price' || param === 'price_per_m') {
                    visParam = 'visible_' + param;
                } else {
                    visParam = param;
                }
                const valElement = resultItemElement.is('[data-' + visParam + ']') ? resultItemElement : resultItemElement.find('[data-' + visParam + ']');

                if (valElement.length) {

                    if (param === 'price' || param === 'price_per_m') {
                        oneResultItem[visParam] = oneResultItem[param].toLocaleString();
                    }
                    switch (valElement.data(visParam)) {
                        case 'text':
                            valElement.text(oneResultItem[visParam]);
                            break;
                        case 'background':
                            valElement.css({
                                backgroundImage: 'url(' + oneResultItem[visParam] + ')'
                            });
                            break;
                        default:
                            valElement.attr(valElement.data(visParam), oneResultItem[visParam]);
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



        const createClusters = (data) => {
            var customItemContentLayout = window['ymaps'].templateLayoutFactory.createClass(
                '<h2 class=ballon_header>{{ properties.balloonContentHeader|raw }}</h2>' +
                '<div class=ballon_body>{{ properties.balloonContentBody|raw }}</div>' +
                '<div class=ballon_footer>{{ properties.balloonContentFooter|raw }}</div>'
            );

            clusterer = new window['ymaps'].Clusterer({
                clusterIcons: [
                    {
                        href: 'static/images/icons/pin_maps.svg',
                        size: [64, 64],
                        offset: [-32, -20]
                    }
                ],
                clusterDisableClickZoom: true,
                clusterOpenBalloonOnClick: true,
                clusterBalloonPanelMaxMapArea: 0,
                clusterBalloonContentLayoutWidth: 500,
                clusterBalloonItemContentLayout: customItemContentLayout,
                clusterBalloonLeftColumnWidth: 220
            });

            myMap.geoObjects.add(clusterer);

            const placemarks = [];

            for (let k in data) {

                const resultItem = data[k];
                searchResult.push(data[k]);
                const pm = new window['ymaps'].Placemark([resultItem.latitude, resultItem.longitude], {
                    balloonContentHeader: data[k].address,
                    balloonContentBody:
                        data[k].price.toLocaleString() + ' ₽<br/>' +
                        '<a href="' + data[k].link + '" target="_blank">' + data[k].link + '</a>'
                    ,
                    balloonContentFooter: '<small style="color: #ddd">Smart Realty</small>'
                }, {
                    iconLayout: 'default#imageWithContent',
                    iconImageHref: 'static/images/icons/pin_maps.svg',
                    iconImageSize: [56, 56],
                    iconImageOffset: [-28, -28],
                    iconContentOffset: [15, 15]
                });

                resultItem.placemark = pm;
                placemarks.push(pm);
            }

            clusterer.add(placemarks);
        };


        const createSearchRequest = () => {
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

        };

        const sendSearchForm = (notCreateData, addData) => {
            if (!notCreateData) {
                createSearchRequest();
            }

            if (addData) {
                formRequest = $.extend(formRequest, addData);
            }

            const resultsBlock = $('#results-list-block');

            resultsBlock.addClass('in-progress');

            const navigate = $('#results-navigate');
            navigate.html('');

            myMap.geoObjects.remove(clusterer);

            $.ajax({
                url: '/api/mean/',
                // url: 'static/test.json',
                data: formRequest
            }).always(() => {
                // results.show();
            }).then((res) => {
                searchResult = res;

                resultsNavBtns = [];
                resultsBlock.show();

                $('#search-items-count').text(searchResult.flats.length);

                for (let n = 0; n < res.max_page; n++) {
                    const navButton = $('<a>');
                    navButton.attr('href', '#results-list');
                    navButton.addClass('results-list_navigation-btn');
                    navButton.text(n + 1);
                    navButton.data('page', n);

                    if (n + 1 === searchResult.page) {
                        navButton.addClass('active');
                    }

                    navButton.on('click', () => {
                        $(this).addClass('active');
                        sendSearchForm(true, {
                            page: n + 1
                        });
                    });

                    resultsNavBtns.push(navButton);
                    navigate.append(navButton);
                }

                resultsBlock.removeClass('in-progress');
                resultsBlock.hide();
                if (searchResult.flats.length) {
                    showResultPage();
                    resultsBlock.show();
                }
            });
        };

        const searchForm = $('#search-form').on('submit', (event) => {
            event.preventDefault();
            sendSearchForm();

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
